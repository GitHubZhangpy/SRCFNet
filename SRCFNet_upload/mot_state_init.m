function im_state = mot_state_init(im_params)

im_state.num_mot_frames = 1;
im_state.frame = 1;                             % 当前帧数
im_state.pos = im_params.pos;                   % 当前帧的中心位置pos
im_state.target_sz = im_params.target_sz;       % 当前size大小
im_state.old_pos = im_params.pos;               % 上一帧的中心位置pos
im_state.old_target_sz = im_params.target_sz;   % 上一帧size大小
im_state.feature = [];                          % 记录当前帧的feature
im_state.hf = [];                               % 记录filter
im_state.xt = [];
% 一些参数需要在调用的时候保持连续性，避免造成undefined情况
im_state.response = [];
im_state.hf_rhs = [];
im_state.hf_autocorr = [];
im_state.hf_vec = [];
im_state.hf_vec_old = [];
im_state.fig_handle = [];

end