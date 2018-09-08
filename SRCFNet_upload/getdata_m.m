function im_data = getdata_m(image,im_params)

% im_data.num_frames = numel(im_params.s_frames);

if im_params.search_area > im_params.filter_max_area
    im_data.currentScaleFactor = sqrt(im_params.search_area / im_params.filter_max_area);
else
    im_data.currentScaleFactor = 1.0;
end

im_data.base_target_sz = im_params.target_sz / im_data.currentScaleFactor;

switch im_params.search_area_shape
    case 'proportional'
        im_data.sz = floor( im_data.base_target_sz * im_params.search_area_scale);
    case 'square'
        im_data.sz = repmat(sqrt(prod(im_data.base_target_sz * im_params.search_area_scale)), 1, 2);
    case 'fix_padding'
        im_data.sz = ...
        im_data.base_target_sz + sqrt(prod(im_data.base_target_sz * im_params.search_area_scale) + (im_data.base_target_sz(1) - im_data.base_target_sz(2))/4) - sum(im_data.base_target_sz)/2;
    otherwise
        error('Unknown "params.search_area_shape". Must be ''proportional'', ''square'' or ''fix_padding''');
end

im_data.sz = round(im_data.sz / im_params.featureRatio) * im_params.featureRatio;
im_data.use_sz = floor(im_data.sz/im_params.featureRatio);

output_sigma = sqrt(prod(floor(im_data.base_target_sz/im_params.featureRatio))) * im_params.output_sigma_factor;
rg = circshift(-floor((im_data.use_sz(1)-1)/2):ceil((im_data.use_sz(1)-1)/2), [0 -floor((im_data.use_sz(1)-1)/2)]);
cg = circshift(-floor((im_data.use_sz(2)-1)/2):ceil((im_data.use_sz(2)-1)/2), [0 -floor((im_data.use_sz(2)-1)/2)]);
[rs, cs] = ndgrid( rg,cg);
im_data.y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = fft2(im_data.y);
im_data.yf_vec = single(yf(:));

if im_params.interpolate_response == 1
    im_data.interp_sz = im_data.use_sz * im_params.featureRatio;
else
    im_data.interp_sz = im_data.use_sz;
end

im_data.cos_window = single(hann(im_data.use_sz(1))*hann(im_data.use_sz(2))');
% maxmax = max(max(im_data.cos_window));
% minmin = min(min(im_data.cos_window));
% mmdiff = maxmax + minmin;
% im_data.cos_window(:,:) = mmdiff - im_data.cos_window(:,:);

im_data.support_sz = prod(im_data.use_sz);

if size(image,3) == 3
    if all(all(image(:,:,1) == image(:,:,2)))
        im_data.colorImage = false;
    else
        im_data.colorImage = true;
    end
else
    im_data.colorImage = false;
end

im_data.feature_dim = 0;
for n = 1:length(im_params.features)
    
    if ~isfield(im_params.features{n}.fparams,'useForColor')
        im_params.features{n}.fparams.useForColor = true;
    end;
    
    if ~isfield(im_params.features{n}.fparams,'useForGray')
        im_params.features{n}.fparams.useForGray = true;
    end;
    
    if (im_params.features{n}.fparams.useForColor && im_data.colorImage) || (im_params.features{n}.fparams.useForGray && ~im_data.colorImage)
        im_data.feature_dim = im_data.feature_dim + im_params.features{n}.fparams.nDim;
    end;
end;

if size(image,3) > 1 && im_data.colorImage == false
    image = image(:,:,1);
end

[im_data.dft_sym_ind, im_data.dft_pos_ind, im_data.dft_neg_ind] = partition_spectrum2(im_data.use_sz);

dfs_sym_ind = (1:length(im_data.dft_sym_ind))';
dfs_real_ind = dfs_sym_ind(end) - 1 + 2 * (1:length(im_data.dft_pos_ind))';
dfs_imag_ind = dfs_sym_ind(end) + 2 * (1:length(im_data.dft_pos_ind))';

im_data.dfs_matrix = dft2dfs_matrix(im_data.dft_sym_ind, im_data.dft_pos_ind, im_data.dft_neg_ind, dfs_sym_ind, dfs_real_ind, dfs_imag_ind);

if im_params.use_reg_window
    
    % create weight window
    im_params.reg_window_min = im_params.ref_window_power;
    
    reg_scale = 0.5 * im_data.base_target_sz/im_params.featureRatio;

    wrg = -(im_data.use_sz(1)-1)/2:(im_data.use_sz(1)-1)/2;
    wcg = -(im_data.use_sz(2)-1)/2:(im_data.use_sz(2)-1)/2;
    [wrs, wcs] = ndgrid(wrg, wcg);
   
    reg_window = (im_params.reg_window_edge - im_params.reg_window_min) * (abs(wrs/reg_scale(1)).^im_params.ref_window_power + abs(wcs/reg_scale(2)).^im_params.ref_window_power) + im_params.reg_window_min;  
    
%     maxmax = max(max(reg_window));
%     minmin = min(min(reg_window));
%     mmdiff = maxmax + minmin;
%     reg_window(:,:) = mmdiff - reg_window(:,:);
    
    % opposite for obtain background's filter
%     tmp_max = max(reg_window(:,1));
%     tmp_mid = (im_data.use_sz(1)-1)/2 + 1;
%     tmp_min = min(reg_window(round(tmp_mid),:));
%     diff = tmp_max + tmp_min;
%     reg_window(:,:) = diff - reg_window(:,:);

    reg_window_dft = fft2(reg_window) / prod(im_data.use_sz);
    reg_window_dft_sep = cat(3, real(reg_window_dft), imag(reg_window_dft));
    reg_window_dft_sep(abs(reg_window_dft_sep) < im_params.reg_sparsity_threshold * max(abs(reg_window_dft_sep(:)))) = 0;
    reg_window_dft = reg_window_dft_sep(:,:,1) + 1i*reg_window_dft_sep(:,:,2);
    
    reg_window_sparse = real(ifft2(reg_window_dft));
    reg_window_dft(1,1) = reg_window_dft(1,1) - im_data.support_sz * min(reg_window_sparse(:)) + im_params.reg_window_min;
    
    regW = cconvmtx2(reg_window_dft);
    
    regW_dfs = real(im_data.dfs_matrix * regW * im_data.dfs_matrix');
    
    im_data.WW_block = regW_dfs' * regW_dfs;
    
    if im_data.support_sz <= 120^2
        im_data.WW_block(0<abs(im_data.WW_block) & abs(im_data.WW_block)<0.00001) = 0;
    end
else
    im_data.WW_block = im_params.lambda * speye(im_data.support_sz);
    im_params.reg_window_min = sqrt(lambda); 
end

WW = eval(['blkdiag(im_data.WW_block' repmat(',im_data.WW_block', 1, im_data.feature_dim-1) ');']);

im_data.WW_L = tril(WW);
im_data.WW_U = triu(WW, 1);

if im_params.nScales > 0
    scale_exp = (-floor((im_params.nScales-1)/2):ceil((im_params.nScales-1)/2));
    
    im_data.scaleFactors = im_params.scale_step .^ scale_exp;
    
    im_data.min_scale_factor = im_params.scale_step ^ ceil(log(max(5 ./ im_data.sz)) / log(im_params.scale_step));
    im_data.max_scale_factor = im_params.scale_step ^ floor(log(min([size(image,1) size(image,2)] ./ im_data.base_target_sz)) / log(im_params.scale_step));
end

im_data.num_sym_coef = length(im_data.dft_sym_ind);

index_i_sym = zeros(2*im_data.feature_dim, length(im_data.dft_sym_ind), im_data.feature_dim);
index_j_sym = zeros(size(index_i_sym));

index_i_sym_re = repmat(bsxfun(@plus, im_data.support_sz*(0:im_data.feature_dim-1)', 1:length(im_data.dft_sym_ind)), [1 1 im_data.feature_dim]);
index_i_sym(1:2:end, :, :) = index_i_sym_re;
index_i_sym(2:2:end, :, :) = NaN;

index_j_sym_re = permute(index_i_sym_re, [3 2 1]);
index_j_sym(1:2:end, :, :) = index_j_sym_re;
index_j_sym(2:2:end, :, :) = NaN;

index_i = zeros(2*im_data.feature_dim, 2*length(im_data.dft_pos_ind), im_data.feature_dim);
index_j = zeros(size(index_i));

index_i_re = repmat(bsxfun(@plus, im_data.support_sz*(0:im_data.feature_dim-1)', (length(im_data.dft_sym_ind)+1:2:im_data.support_sz)), [1 1 im_data.feature_dim]);
index_i(1:2:end, 1:2:end, :) = index_i_re;
index_i(2:2:end, 1:2:end, :) = index_i_re + 1;
index_i(1:2:end, 2:2:end, :) = index_i_re;
index_i(2:2:end, 2:2:end, :) = index_i_re + 1;

index_j_re = permute(index_i_re, [3 2 1]);
index_j(1:2:end, 1:2:end, :) = index_j_re;
index_j(2:2:end, 1:2:end, :) = index_j_re;
index_j(1:2:end, 2:2:end, :) = index_j_re + 1;
index_j(2:2:end, 2:2:end, :) = index_j_re + 1;

index_i = cat(2, index_i_sym, index_i);
index_j = cat(2, index_j_sym, index_j);

index_i = index_i(:);
index_j = index_j(:);

zero_ind = (index_i == index_j-1) | (index_i == index_j+1);
index_i(zero_ind) = NaN;
index_j(zero_ind) = NaN;

im_data.data_L_mask = index_i >= index_j;
im_data.data_U_mask = index_i < index_j;

data_L_i = index_i(im_data.data_L_mask);
data_L_j = index_j(im_data.data_L_mask);
data_U_i = index_i(im_data.data_U_mask);
data_U_j = index_j(im_data.data_U_mask);

WW_L_ind = find(im_data.WW_L);
data_L_ind = sub2ind(size(im_data.WW_L), data_L_i, data_L_j);

[L_ind, ~, data_WW_in_L_index] = unique([data_L_ind; WW_L_ind]);

im_data.data_in_L_index = uint32(data_WW_in_L_index(1:length(data_L_ind)));
WW_in_L_index = data_WW_in_L_index(length(data_L_ind)+1:end);

nnz_L = length(L_ind);
WW_L_vec = zeros(nnz_L, 1, 'single');
WW_L_vec(WW_in_L_index) = full(im_data.WW_L(WW_L_ind));

im_data.WW_L_vec_data = WW_L_vec(im_data.data_in_L_index);

im_data.L_vec = WW_L_vec;

mat_size = im_data.feature_dim * im_data.support_sz;
[L_i, L_j] = ind2sub(size(im_data.WW_L), L_ind);
im_data.AL = sparse(L_i, L_j, ones(nnz_L,1), mat_size, mat_size);
im_data.AU_data = sparse(data_U_i, data_U_j, ones(length(data_U_i),1), mat_size, mat_size);

if im_params.interpolate_response >= 3
    im_data.ky = circshift(-floor((im_data.use_sz(1) - 1)/2) : ceil((im_data.use_sz(1) - 1)/2), [1, -floor((im_data.use_sz(1) - 1)/2)]);
    im_data.kx = circshift(-floor((im_data.use_sz(2) - 1)/2) : ceil((im_data.use_sz(2) - 1)/2), [1, -floor((im_data.use_sz(2) - 1)/2)])';
%%% gradient_ascent_iterations = params.gradient_ascent_iterations;
%%% gradient_step_size = single(params.gradient_step_size);
end

im_data.xxlf_sep = zeros(2*im_data.feature_dim, length(im_data.dft_sym_ind) + 2 * length(im_data.dft_pos_ind), im_data.feature_dim, 'single');
im_data.multires_pixel_template = zeros(im_data.sz(1), im_data.sz(2), size(image,3), im_params.nScales, 'uint8');

end