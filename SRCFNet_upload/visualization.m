function [ im_state ] = visualization(im_state, im, frame, im_data, im_params)
 
if im_params.visualization == 1
    rect_position_vis = [im_state.pos([2,1]) - im_state.target_sz([2,1])/2, im_state.target_sz([2,1])];
    im_to_show = double(im)/255;
    if size(im_to_show,3) == 1
        im_to_show = repmat(im_to_show, [1 1 3]);
    end
    if frame == 1
        im_state.fig_handle = figure('Name', 'Tracking');
        imagesc(im_to_show);
        hold on;
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, int2str(frame), 'color', [0 1 1]);
        hold off;
        axis off;axis image;set(gca, 'Units', 'normalized', 'Position', [0 0 1 1])
    else
        resp_sz = round(im_data.sz*im_data.currentScaleFactor*im_data.scaleFactors(im_params.nScales));
        xs = floor(im_state.old_pos(2)) + (1:resp_sz(2)) - floor(resp_sz(2)/2);
        ys = floor(im_state.old_pos(1)) + (1:resp_sz(1)) - floor(resp_sz(1)/2);
        sc_ind = floor((im_params.nScales - 1)/2) + 1;
        
        figure(im_state.fig_handle);
        imagesc(im_to_show);
        hold on;
        resp_handle = imagesc(xs, ys, fftshift(im_state.response(:,:,sc_ind))); colormap hsv;
        alpha(resp_handle, 0.5);
        rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth',2);
        text(10, 10, int2str(frame), 'color', [0 1 1]);
        hold off;
    end
    drawnow
end
end
