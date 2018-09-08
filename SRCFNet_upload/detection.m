function [isuse_bg, state, data] = detection(params,data,state,im)

state.old_pos = inf(size(state.pos));
state.old_target_sz = state.target_sz;
iter = 1;

while iter <= params.refinement_iterations && any(state.old_pos ~= state.pos)
    for scale_ind = 1:params.nScales
        data.multires_pixel_template(:,:,:,scale_ind) = ...
            get_pixels(im, state.pos, round(data.sz*data.currentScaleFactor*data.scaleFactors(scale_ind)), data.sz);
    end
    
    xt = bsxfun(@times,get_features(data.multires_pixel_template,params.features,params.global_feat_params),data.cos_window);
    state.xt = xt;
    xtf = fft2(xt);
    
    responsef = permute(sum(bsxfun(@times, state.hf, xtf), 3), [1 2 4 3]);
    
    if params.interpolate_response == 2
        data.interp_sz = floor(size(data.y) * params.featureRatio * data.currentScaleFactor);
    end
    responsef_padded = resizeDFT2(responsef, data.interp_sz);
    
    state.response = ifft2(responsef_padded, 'symmetric');
    
    % find maximum
    if params.interpolate_response == 3
        error('Invalid parameter value for interpolate_response');
    elseif params.interpolate_response == 4
        [max_scale_response,disp_row, disp_col, sind] = resp_newton1(state.response, responsef_padded, params.newton_iterations, data.ky, data.kx, data.use_sz);
    else
        [row, col, sind] = ind2sub(size(state.response), find(state.response == max(state.response(:)), 1));
        disp_row = mod(row - 1 + floor((data.interp_sz(1)-1)/2), data.interp_sz(1)) - floor((data.interp_sz(1)-1)/2);
        disp_col = mod(col - 1 + floor((data.interp_sz(2)-1)/2), data.interp_sz(2)) - floor((data.interp_sz(2)-1)/2);
    end
    
    isuse_bg = 0;

    % judge whether use bg info or not
    overlap_pred = pred_overlap(state.reg.model, max_scale_response);
    if state.frame == 2
        state.max_response = overlap_pred;
    end
    % /state.max_response

    if overlap_pred/state.max_response < 0.5
        isuse_bg = 0;
        return;
    end
    
%     disp([state.frame, max_scale_response, overlap_pred,overlap_pred/state.max_response]);
    
    switch params.interpolate_response
        case 0
            translation_vec = round([disp_row, disp_col] * params.featureRatio * data.currentScaleFactor * data.scaleFactors(sind));
        case 1
            translation_vec = round([disp_row, disp_col] * data.currentScaleFactor * data.scaleFactors(sind));
        case 2
            translation_vec = round([disp_row, disp_col] * data.scaleFactors(sind));
        case 3
            translation_vec = round([disp_row, disp_col] * params.featureRatio * data.currentScaleFactor * data.scaleFactors(sind));
        case 4
            translation_vec = round([disp_row, disp_col] * params.featureRatio * data.currentScaleFactor * data.scaleFactors(sind));
    end
    
        data.currentScaleFactor = data.currentScaleFactor * data.scaleFactors(sind);
        % adjust to make sure we are not to large or to small
        if data.currentScaleFactor < data.min_scale_factor
            data.currentScaleFactor = data.min_scale_factor;
        elseif data.currentScaleFactor > data.max_scale_factor
            data.currentScaleFactor = data.max_scale_factor;
        end
    
    % update position
    state.old_pos = state.pos;
    state.pos = state.pos + translation_vec;
    iter = iter + 1;
end

% if params.debug
%     figure(101);
%     subplot_cols = ceil(sqrt(params.nScales));
%     subplot_rows = ceil(params.nScales/subplot_cols);
%     for scale_ind = 1:params.nScales
%         subplot(subplot_rows,subplot_cols,scale_ind);
%         imagesc(fftshift(state.response(:,:,scale_ind)));colorbar; axis image;
%         title(sprintf('Scale %i,  max(response) = %f', scale_ind, max(max(params.response(:,:,scale_ind)))));
%     end
% end

end



















% function [isuse_bg, state] = detection(params,data,state,im)
% 
%         iter = 1;
%         state.old_pos = inf(size(state.pos));
%         state.old_target_sz = state.target_sz;
%         %translation search
%         while iter <= refinement_iterations && any(old_pos ~= pos)
%             % Get multi-resolution image
%             for scale_ind = 1:nScales
%                 multires_pixel_template(:,:,:,scale_ind) = ...
%                     get_pixels(im, pos, round(sz*currentScaleFactor*scaleFactors(scale_ind)), sz);
%             end
%             
%             xt = bsxfun(@times,get_features(multires_pixel_template,features,global_feat_params),cos_window);
%             
%             xtf = fft2(xt);
%             
%             responsef = permute(sum(bsxfun(@times, hf, xtf), 3), [1 2 4 3]);
%             
%             % if we undersampled features, we want to interpolate the
%             % response so it has the same size as the image patch
%             if interpolate_response == 2
%                 % use dynamic interp size
%                 interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
%             end
%             responsef_padded = resizeDFT2(responsef, interp_sz);
%             
%             % response
%             response = ifft2(responsef_padded, 'symmetric');
%             
%             % find maximum
%             if interpolate_response == 3
%                 [disp_row, disp_col, sind] = resp_gradient_ascent(response, responsef_padded, gradient_ascent_iterations, gradient_step_size, ky, kx, use_sz);
%             elseif interpolate_response == 4
%                 [disp_row, disp_col, sind] = resp_newton(response, responsef_padded, gradient_ascent_iterations, ky, kx, use_sz);
%             else
%                 [row, col, sind] = ind2sub(size(response), find(response == max(response(:)), 1));
%                 disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
%                 disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
%             end
%             
%             % calculate translation
%             switch interpolate_response
%                 case 0
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%                 case 1
%                     translation_vec = round([disp_row, disp_col] * currentScaleFactor * scaleFactors(sind));
%                 case 2
%                     translation_vec = round([disp_row, disp_col] * scaleFactors(sind));
%                 case 3
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%                 case 4
%                     translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor * scaleFactors(sind));
%             end
%             
%             % set the scale
%             currentScaleFactor = currentScaleFactor * scaleFactors(sind);
%             % adjust to make sure we are not to large or to small
%             if currentScaleFactor < min_scale_factor
%                 currentScaleFactor = min_scale_factor;
%             elseif currentScaleFactor > max_scale_factor
%                 currentScaleFactor = max_scale_factor;
%             end
%             
%             % update position
%             old_pos = pos;
%             pos = pos + translation_vec;
%             
%             iter = iter + 1;
%         end
%         
%         % debug visualization
% %         if debug
% %             figure(101);
% %             subplot_cols = ceil(sqrt(nScales));
% %             subplot_rows = ceil(nScales/subplot_cols);
% %             for scale_ind = 1:nScales
% %                 subplot(subplot_rows,subplot_cols,scale_ind);
% %                 imagesc(fftshift(response(:,:,scale_ind)));colorbar; axis image;
% %                 title(sprintf('Scale %i,  max(response) = %f', scale_ind, max(max(response(:,:,scale_ind)))));
% %             end
% %         end
% 
% end