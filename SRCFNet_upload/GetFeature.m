function im_state = GetFeature(fr, image, im_state, im_data, im_params)

im_state.frame = fr;
im_state.old_pos = im_state.pos;
im_state.old_target_sz = im_state.target_sz;
pixels = get_pixels(image,im_state.pos,round(im_data.sz*im_data.currentScaleFactor),im_data.sz);
xl = bsxfun(@times,get_features(pixels,im_params.features,im_params.global_feat_params),im_data.cos_window);
im_state.xt = xl;
xlf = fft2(xl);
im_state.feature = reshape(xlf, [im_data.support_sz, im_data.feature_dim]);

end