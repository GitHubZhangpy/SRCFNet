function [ deep_feature ] = get_deep_feature( im, fparam, gparam )
nScale = 7; % scale number 
fsize = 109; % feature hight and width 
area = fsize^2; % feature area
feature_dim = 96; % feature dim before pca 

fparam.net.mode = 'test';
im = single(im);
im = imresize(im, fparam.net.meta.normalization.imageSize(1:2));
im = bsxfun(@minus, im, fparam.net.meta.normalization.averageImage);
im = gpuArray(im);
fparam.net.conserveMemory = 0;
fparam.net.eval({'x0',im});
deep_feature = fparam.net.vars(fparam.net.getVarIndex('x3')).value; %% x14
deep_feature = squeeze(gather(deep_feature));
deep_feature_temp= zeros(area,fparam.nDim);
deep_feature_reshape = zeros(fsize,fsize,fparam.nDim,nScale);
if size(size(deep_feature),2) == 3
    deep_feature = reshape(deep_feature,[area,feature_dim]);
    [~,score] = princomp(deep_feature);
    deep_feature = score(:,1:fparam.nDim);
    deep_feature = reshape(deep_feature,[fsize,fsize,fparam.nDim]);
else
    deep_feature = reshape(deep_feature,[area,feature_dim,size(deep_feature,4)]);
    for i=1:size(deep_feature,3)
        [~,score] = princomp(deep_feature(:,:,i));
        deep_feature_temp(:,:,i)= score(:,1:fparam.nDim);
        deep_feature_reshape(:,:,:,i) = reshape(deep_feature_temp(:,:,i),[fsize,fsize,fparam.nDim]);
    end
        deep_feature = deep_feature_reshape;
end




