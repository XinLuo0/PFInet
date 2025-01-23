clear; close all;

%% path
savepath = './Training_mat_5x5/1/';
if exist(savepath, 'dir')==0
    mkdir(savepath);
end

sourceDataPath = './L3F-dataset/jpeg/train/1/'; % path to L3F-dataset
sourceScenes = dir(sourceDataPath);
sourceScenes(1:2) = [];
sceneNum = length(sourceScenes);
%% params
H = 433;
W = 625;

allah = 15;
allaw = 15;

ah = 5;
aw = 5;
an_crop = ceil((allah - ah) / 2)+1;
for iScene = 1 : sceneNum
    sceneName = sourceScenes(iScene).name;
    dataPath = [sourceDataPath, sceneName];
    lfimage = single(imread(dataPath))/255.0;
    img = zeros(allah,allaw,H,W-4,3,'single');
    for v = 1 : allah
        for u = 1 : allah            
            sub = lfimage((v-1)*H+1:v*H,(u-1)*W+1:u*W,:);
            img(v,u,:,:,:) = sub(:,3:end-2,:);        
        end
    end
    LF = img(an_crop:ah+an_crop-1,an_crop:aw+an_crop-1,:,:,:);
    size(LF)
    save_path = [savepath, sceneName(1:end-4), '.mat'];
    save(save_path, 'LF'); 
    
end