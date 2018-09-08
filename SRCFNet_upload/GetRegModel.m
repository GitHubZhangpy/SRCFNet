function im_state = GetRegModel(im_state, im_data)

xtf = fft2(im_state.xt);
responsef = permute(sum(bsxfun(@times, im_state.hf, xtf), 3), [1 2 4 3]);
responsef_padded = resizeDFT2(responsef, im_data.interp_sz);
im_state.response = ifft2(responsef_padded, 'symmetric');
resp_shift = fftshift(im_state.response(:,:,1));
reg = jud_overlap([im_state.pos([2,1]), im_state.target_sz([2,1])], resp_shift);
im_state.reg = reg;

end