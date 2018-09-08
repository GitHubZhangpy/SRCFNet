function [hf_out] = net_update(xxf,xyf,hf,use_sz,base_target_sz)

maxit = 7;
thf = hf;
it = 1;
lr = 1.5e-5;
threshold = 10000;
%% wd
ref_window_power = 5;
reg_window_edge = 3;
reg_window_min = 0.1;
reg_scale = 0.5 * base_target_sz/4;
wrg = -(use_sz(1)-1)/2:(use_sz(1)-1)/2;
wcg = -(use_sz(2)-1)/2:(use_sz(2)-1)/2;
[wrs, wcs] = ndgrid(wrg, wcg);
wd = (reg_window_edge - reg_window_min) * (abs(wrs/reg_scale(1)).^ref_window_power + abs(wcs/reg_scale(2)).^ref_window_power) + reg_window_min;
wd = wd .* wd;
wd(wd>threshold) = threshold;
wd = wd .* lr ;
wd = 1 - wd;
while it<maxit
    gradf =  bsxfun(@times, xxf, thf)-xyf;
    grad = real(ifft2(gradf));
    thf_time = ifft2(thf);
    thf_time = bsxfun(@times,wd,thf_time)-lr.*grad ;
    thf = fft2(thf_time);
    it = it+1;
end

hf_out = thf;