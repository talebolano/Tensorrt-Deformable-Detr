import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.onnx.symbolic_helper import parse_args
from torch.onnx import register_custom_op_symbolic
from onnxsim import simplify
from mmdet.apis import init_detector
import argparse
from mmdet.models.utils.transformer import inverse_sigmoid
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from types import MethodType
from mmdet.core import multi_apply


class Etmpy_MultiScaleDeformableAttnFunction(torch.autograd.Function):
    @staticmethod
    def symbolic(g,value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):

        return g.op('com.microsoft::MultiscaleDeformableAttnPlugin_TRT',value, value_spatial_shapes, value_level_start_index,
                    sampling_locations, attention_weights)
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        '''
        no real mean,just for inference
        '''
        bs, _, mum_heads, embed_dims_num_heads = value.shape
        bs ,num_queries, _, _, _, _ = sampling_locations.shape
        return value.new_zeros(bs, num_queries, mum_heads, embed_dims_num_heads)

    @staticmethod
    def backward(ctx, grad_output):
        pass                


def MSMHDA_onnx_export(self,
            query,
            key=None,
            value=None,
            identity=None,
            query_pos=None,
            key_padding_mask=None,
            reference_points=None,
            spatial_shapes=None,
            level_start_index=None,
            **kwargs):

    if value is None:
        value = query

    if identity is None:
        identity = query
    if query_pos is not None:
        query = query + query_pos
    if not self.batch_first:
        # change to (bs, num_query ,embed_dims)
        query = query.permute(1, 0, 2)
        value = value.permute(1, 0, 2)

    bs, num_query, _ = query.shape
    bs, num_value, _ = value.shape
    assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

    value = self.value_proj(value)
    if key_padding_mask is not None:
        value = value.masked_fill(key_padding_mask[..., None], 0.0)
    value = value.view(int(bs), int(num_value), self.num_heads, -1)
    sampling_offsets = self.sampling_offsets(query).view(
        int(bs), int(num_query), self.num_heads, self.num_levels, self.num_points, 2)
    attention_weights = self.attention_weights(query).view(
        int(bs), int(num_query), self.num_heads, self.num_levels * self.num_points)
    attention_weights = attention_weights.softmax(-1)

    attention_weights = attention_weights.view(int(bs), int(num_query),
                                                self.num_heads,
                                                self.num_levels,
                                                self.num_points)
    if reference_points.shape[-1] == 2:
        offset_normalizer = torch.stack(
            [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
    elif reference_points.shape[-1] == 4:
        sampling_locations = reference_points[:, :, None, :, None, :2] \
            + sampling_offsets / self.num_points \
            * reference_points[:, :, None, :, None, 2:] \
            * 0.5
    else:
        raise ValueError(
            f'Last dim of reference_points must be'
            f' 2 or 4, but get {reference_points.shape[-1]} instead.')

    output = Etmpy_MultiScaleDeformableAttnFunction.apply(
        value, spatial_shapes, level_start_index, sampling_locations,
        attention_weights, self.im2col_step)

    output = output.reshape(int(bs),int(num_query),self.embed_dims)
    output = self.output_proj(output)
    output = output.reshape(int(bs),int(num_query),self.embed_dims)

    if not self.batch_first:
        # (num_query, bs ,embed_dims)
        output = output.permute(1, 0, 2)

    return self.dropout(output) + identity


@parse_args('v','v','i','i','b')
def grid_sampler(g, input, grid, mode_enum, padding_mode_enum, align_corners):
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode_enum]
    padding_str = ['zeros', 'border', 'reflection'][padding_mode_enum]
    return g.op('com.microsoft::GridSample',input,grid,mode_s=mode_str,padding_mode_s=padding_str,align_corners_i=align_corners)


def deformable_detr_head_onnx_export(self, mlvl_feats):

    batch_size = mlvl_feats[0].size(0)
    img_masks = mlvl_feats[0].new_zeros(
        (batch_size, mlvl_feats[0].size(2), mlvl_feats[0].size(3)))

    mlvl_masks = []
    mlvl_positional_encodings = []
    for feat in mlvl_feats:
        mlvl_masks.append(
            F.interpolate(img_masks[None],
                            size=feat.shape[-2:]).to(torch.bool).squeeze(0))
        mlvl_positional_encodings.append(
            self.positional_encoding(mlvl_masks[-1]))

    query_embeds = None
    if not self.as_two_stage:
        query_embeds = self.query_embedding.weight
    hs, init_reference, inter_references, \
        enc_outputs_class, enc_outputs_coord = self.transformer(
                mlvl_feats,
                mlvl_masks,
                query_embeds,
                mlvl_positional_encodings,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None  # noqa:E501
        )
    hs = hs.permute(1,0,2)
    lvl = 5
    reference = inter_references#[lvl - 1]
    reference = inverse_sigmoid(reference)
    outputs_class = self.cls_branches[lvl](hs)
    tmp = self.reg_branches[lvl](hs)
    if reference.shape[-1] == 4:
        tmp += reference
    else:
        assert reference.shape[-1] == 2
        tmp[..., :2] += reference
    outputs_coord = tmp.sigmoid()
 
    return outputs_class, outputs_coord


def deformable_detr_transformer_gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                    spatial_shapes):

    N, S, C = memory.shape
    proposals = []
    _cur = 0
    for lvl, (H, W) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
            N, H, W, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(
                0, H - 1, H, dtype=torch.float32, device=memory.device),
            torch.linspace(
                0, W - 1, W, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1),
                            valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
        proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
        proposals.append(proposal)
        _cur += (H * W)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (((output_proposals > 0.01) &
                                (output_proposals < 0.99)).sum(-1) ==4).unsqueeze(-1)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                float(0))
    output_memory = self.enc_output_norm(self.enc_output(output_memory))
    return output_memory, output_proposals


def deformable_detr_transformer_onnx_export(self,
            mlvl_feats,
            mlvl_masks,
            query_embed,
            mlvl_pos_embeds,
            reg_branches=None,
            cls_branches=None,
            **kwargs):

    def gather_index_single(feats,index):
        # n,4 300
        return feats[index,:].unsqueeze(0),None

    assert self.as_two_stage or query_embed is not None

    feat_flatten = []
    mask_flatten = []
    lvl_pos_embed_flatten = []
    spatial_shapes = []
    for lvl, (feat, mask, pos_embed) in enumerate(
            zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
        bs, c, h, w = feat.shape
        spatial_shape = (h, w)
        spatial_shapes.append(spatial_shape)
        feat = feat.flatten(2).transpose(1, 2) # b,c,hw --> b,hw,c
        mask = mask.flatten(1)
        pos_embed = pos_embed.flatten(2).transpose(1, 2)
        lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
        lvl_pos_embed_flatten.append(lvl_pos_embed)
        feat_flatten.append(feat)
        mask_flatten.append(mask)
    feat_flatten = torch.cat(feat_flatten, 1) # b,hw,c
    mask_flatten = torch.cat(mask_flatten, 1)
    lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
    spatial_shapes = torch.as_tensor(
        spatial_shapes, dtype=torch.long, device=feat_flatten.device) #  (),(),()
    level_start_index = torch.cat((spatial_shapes.new_zeros(
        (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
    valid_ratios = torch.stack(
        [self.get_valid_ratio(m) for m in mlvl_masks], 1) # 

    reference_points = \
        self.get_reference_points(spatial_shapes,
                                    valid_ratios,
                                    device=feat.device)

    feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, embed_dims)
    lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(
        1, 0, 2)  # (H*W, bs, embed_dims)
    memory = self.encoder(
        query=feat_flatten,
        key=None,
        value=None,
        query_pos=lvl_pos_embed_flatten,
        query_key_padding_mask=mask_flatten,
        spatial_shapes=spatial_shapes,
        reference_points=reference_points,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        **kwargs)

    memory = memory.permute(1, 0, 2)
    bs, _, c = memory.shape
    if self.as_two_stage:
        output_memory, output_proposals = \
            self.gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes)
        enc_outputs_class = cls_branches[self.decoder.num_layers](
            output_memory)
        enc_outputs_coord_unact = \
            reg_branches[
                self.decoder.num_layers](output_memory) + output_proposals

        topk = self.two_stage_num_proposals

        topk_proposals = torch.topk(
            enc_outputs_class[..., 0:1], topk, dim=1)[1].squeeze(2) # 1,300

        topk_coords_unact = torch.cat(multi_apply(gather_index_single,enc_outputs_coord_unact,topk_proposals)[0],dim=0)

        topk_coords_unact = topk_coords_unact.detach()
        reference_points = topk_coords_unact.sigmoid()
        init_reference_out = reference_points
        pos_trans_out = self.pos_trans_norm(
            self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
        query_pos, query = torch.split(pos_trans_out, c, dim=2)
    else:
        query_pos, query = torch.split(query_embed, c, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()
        init_reference_out = reference_points

    # decoder
    query = query.permute(1, 0, 2)
    memory = memory.permute(1, 0, 2)
    query_pos = query_pos.permute(1, 0, 2)
    inter_states, inter_references = self.decoder(
        query=query,
        key=None,
        value=memory,
        query_pos=query_pos,
        key_padding_mask=mask_flatten,
        reference_points=reference_points,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        valid_ratios=valid_ratios,
        reg_branches=reg_branches,
        **kwargs)

    inter_references_out = inter_references
    if self.as_two_stage:
        return inter_states, init_reference_out,\
            inter_references_out, enc_outputs_class,\
            enc_outputs_coord_unact
    return inter_states, init_reference_out, \
        inter_references_out, None, None


def deformable_detr_transformer_decoder_onnx_export(self,
                query,
                *args,
                reference_points=None,
                valid_ratios=None,
                reg_branches=None,
                **kwargs):

    output = query
    intermediate = []
    for lid, layer in enumerate(self.layers):
        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * \
                torch.cat([valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * \
                valid_ratios[:, None]
        output = layer(
            output,
            *args,
            reference_points=reference_points_input,
            **kwargs)
        output = output.permute(1, 0, 2)

        if reg_branches is not None:
            tmp = reg_branches[lid](output)
            if reference_points.shape[-1] == 4:
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points)
                new_reference_points = new_reference_points.sigmoid()
            else:
                assert reference_points.shape[-1] == 2
                new_reference_points = tmp
                new_reference_points[..., :2] = tmp[
                    ..., :2] + inverse_sigmoid(reference_points)
                new_reference_points = new_reference_points.sigmoid()
            reference_points = new_reference_points.detach()

        output = output.permute(1, 0, 2)

    return output, reference_points


def detr_onnx_export(self,img):
    x = self.extract_feat(img) # level_feats

    det_labels,det_bboxes = self.bbox_head(x)

    return det_labels,det_bboxes


def parse():
    opt = argparse.ArgumentParser()
    opt.add_argument("--h",type=int)
    opt.add_argument("--w",type=int)
    opt.add_argument("--config",type=str)
    opt.add_argument("--checkpoint",type=str)
    opt.add_argument("--output",type=str)
    return opt.parse_args()


if __name__=="__main__":
    opt = parse()

    opsetversion=11
    register_custom_op_symbolic("::grid_sampler", grid_sampler, opsetversion)

    model = init_detector(opt.config,
                        opt.checkpoint,
                        device='cpu')
    for moudle in model.modules():
        if isinstance(moudle,MultiScaleDeformableAttention):
            moudle.forward = MethodType(MSMHDA_onnx_export,moudle)

    model.eval()
    model.forward = model.onnx_export

    model.forward = MethodType(detr_onnx_export,model)
    model.bbox_head.forward = MethodType(deformable_detr_head_onnx_export,model.bbox_head)
    model.bbox_head.transformer.gen_encoder_output_proposals = MethodType(deformable_detr_transformer_gen_encoder_output_proposals,model.bbox_head.transformer)
    model.bbox_head.transformer.forward = MethodType(deformable_detr_transformer_onnx_export,model.bbox_head.transformer)
    model.bbox_head.transformer.decoder.forward = MethodType(deformable_detr_transformer_decoder_onnx_export,model.bbox_head.transformer.decoder)

    x = torch.randn(1,3,opt.h,opt.w)#.cuda()

    torch.onnx.export(model,x,opt.output,verbose=True,
                enable_onnx_checker=False,
                operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
                opset_version=opsetversion,
                # custom_opsets={"MultiscaleDeformableAttnPlugin_TRT":1},
                )

    model_simple,check = simplify(opt.output,)
    assert check, "Failed to simplify ONNX model."

    onnx.save(model_simple,opt.output)