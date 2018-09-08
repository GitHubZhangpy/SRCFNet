function im_state = mot_state_init(im_params)

im_state.num_mot_frames = 1;
im_state.frame = 1;                             % ��ǰ֡��
im_state.pos = im_params.pos;                   % ��ǰ֡������λ��pos
im_state.target_sz = im_params.target_sz;       % ��ǰsize��С
im_state.old_pos = im_params.pos;               % ��һ֡������λ��pos
im_state.old_target_sz = im_params.target_sz;   % ��һ֡size��С
im_state.feature = [];                          % ��¼��ǰ֡��feature
im_state.hf = [];                               % ��¼filter
im_state.xt = [];
% һЩ������Ҫ�ڵ��õ�ʱ�򱣳������ԣ��������undefined���
im_state.response = [];
im_state.hf_rhs = [];
im_state.hf_autocorr = [];
im_state.hf_vec = [];
im_state.hf_vec_old = [];
im_state.fig_handle = [];

end