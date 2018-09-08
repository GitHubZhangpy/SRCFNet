function params = setparams(region,im)

params.debug = 0;
params.filter_max_area = 200^2;


deep_params.nDim = 40 ;
net = dagnn.DagNN.loadobj(load('E:\correlation_filter\SRDCF\vot-toolkit\trackers\SRDCF_VOT\imagenet-vgg-m-2048.mat'));
deep_params.net = net;
params.features = {
    struct('getFeature',@get_deep_feature,'fparams',deep_params)
};

params.interpolate_response = 4;
params.init_strategy =  'indep';
params.learning_rate = 0.025;
params.lambda = 1;
params.nScales = 7;
params.newton_iterations = 5;
params.num_GS_iter = 4;
params.output_sigma_factor = 1/16;
params.mu_max = 20;
params.beta = 1.2;
params.eta = 0.025;


%%%
bb_scale = 1;
% If the provided region is a polygon ...
if numel(region) > 4
    % Init with an axis aligned bounding box with correct area and center
    % coordinate
    cx = mean(region(1:2:end));
    cy = mean(region(2:2:end));
    x1 = min(region(1:2:end));
    x2 = max(region(1:2:end));
    y1 = min(region(2:2:end));
    y2 = max(region(2:2:end));
    A1 = norm(region(1:2) - region(3:4)) * norm(region(3:4) - region(5:6));
    A2 = (x2 - x1) * (y2 - y1);
    s = sqrt(A1/A2);
    w = s * (x2 - x1) + 1;
    h = s * (y2 - y1) + 1;
else
    cx = region(1) + (region(3) - 1)/2;
    cy = region(2) + (region(4) - 1)/2;
    w = region(3);
    h = region(4);
end

init_c = [cx cy];
init_sz = bb_scale * [w h];

im_size = size(im);
im_size = im_size([2 1]);

init_pos = min(max(round(init_c - (init_sz - 1)/2), [1 1]), im_size);
init_sz = min(max(round(init_sz), [1 1]), im_size - init_pos + 1);
%%%
params.pos = init_pos+(init_sz-1)/2;
params.pos = [params.pos(2), params.pos(1)];
params.refinement_iterations = 1;
params.ref_window_power = 2;
params.reg_window_min = 0.1;
params.reg_sparsity_threshold = 0.05;
params.reg_window_edge = 3.0; 
params.search_area_scale = 3.5;  %%changed
 
params.search_area_shape = 'square';
params.scale_step = 1.01;
% params.target_sz = floor([region(4),region(3)]);
params.target_sz = [init_sz(2), init_sz(1)];
params.use_reg_window = 1;
params.visualization = 0;
%

params.init_target_sz = params.target_sz;
params.search_area = prod(params.init_target_sz * params.search_area_scale);

if params.search_area > params.filter_max_area
    params.currentScaleFactor = sqrt(params.search_area / params.filter_max_area);
else
    params.currentScaleFactor = 1.0;
end

params.base_target_sz = params.target_sz / params.currentScaleFactor;

switch params.search_area_shape
    case 'proportional'
        params.sz = floor( params.base_target_sz * params.search_area_scale);
    case 'square'
        params.sz = repmat(sqrt(prod(params.base_target_sz * params.search_area_scale)), 1, 2);
    case 'fix_padding'
        params.sz = ...
        params.base_target_sz + sqrt(prod(params.base_target_sz * params.search_area_scale) + (params.base_target_sz(1) - params.base_target_sz(2))/4) - sum(params.base_target_sz)/2;
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end
%im_data.sz = round(im_data.sz / im_params.featureRatio) * im_params.featureRatio;
% im_data.use_sz = floor(im_data.sz/im_params.featureRatio);
params.sz = single(params.sz);
params.use_sz = [109,109];
params.featureRatio = params.sz(1)/params.use_sz(1);

params.t_global.cell_size = params.featureRatio; 
%%
% params.t_global.cell_selection_thresh = 0.75^2; 
% 
% if isfield(params.t_global, 'cell_selection_thresh')
%     if params.search_area < params.t_global.cell_selection_thresh * params.filter_max_area
%         params.t_global.cell_size = ...
%         min(params.featureRatio, max(1, ceil(sqrt(prod(params.init_target_sz * params.search_area_scale)/(params.t_global.cell_selection_thresh * params.filter_max_area / 16)))));
%         
%         params.featureRatio = params.t_global.cell_size;
%         params.search_area = prod(params.init_target_sz / params.featureRatio * params.search_area_scale);
%     end
% end
%%%
params.global_feat_params = params.t_global;

end