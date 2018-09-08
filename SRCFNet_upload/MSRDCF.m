function MSRDCF
% cleanup = onCleanup(@() exit() );
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% VOT: Get initialization data 数据的初始化
[handle, image_path, region] = vot('polygon');
image = imread(image_path);

% initialization
im_params = setparams(region,image);
im_params.visualization = 1;
im_data = getdata(image,im_params);
im_state = state_init(im_params);
use_bg = 0;
init_bg = 0;
mot_state = [];
mot_params = [];
mot_data = [];


% image level tracker init
if size(image,3) > 1 && im_data.colorImage == false
    image = image(:,:,1);
end

fr = 1;
im_state = GetFeature(fr, image, im_state, im_data, im_params);
%!!!!!!im_state = UpdateFilter(im_state,im_params,im_data);
[im_state,im_data] = UpdateFilter(im_state,im_params,im_data);

im_state = visualization(im_state, image, fr, im_data, im_params);
im_state = GetRegModel(im_state, im_data);

while true

    fr = fr + 1;
    [handle, image_path] = handle.frame(handle);
    if isempty(image_path)
        break;
    end
    image = imread(image_path);
    im_state.frame = fr;
    
    if size(image,3) > 1 && im_data.colorImage == false
        image = image(:,:,1);
    end
    
    [use_bg, init_bg, im_state, mot_params, mot_data, mot_state,im_data] = new_detection(use_bg, init_bg, im_params, im_data, im_state, image, mot_state, mot_params, mot_data);
    
    nregion(1) = double(im_state.pos(2) - im_state.target_sz(2)/2);
    nregion(2) = double(im_state.pos(1) - im_state.target_sz(1)/2);
    nregion(3) = double(im_state.target_sz(2));
    nregion(4) = double(im_state.target_sz(1));
    nregion(find(nregion<0)) = 1;
    
    im_state = visualization(im_state, image, fr, im_data, im_params); 

    handle = handle.report(handle, nregion);

end

handle.quit(handle);


end