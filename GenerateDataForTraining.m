%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting

angRes = 5;
patchsize = 64;
stride = 48;
level = 100;
sourceDataPath = './Training_mat_5x5/';
sourceDatasets = dir(sourceDataPath);%获得指定文件夹下的所有子文件夹和文件
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);

SavePath = ['./5x5Dataset/TrainingData_', num2str(level), '_', 'LLE', '_', num2str(angRes), 'x', num2str(angRes), '/'];
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end
inputdir = ['/1_', num2str(level), '/'];
GTFolder = [sourceDataPath, '/1/'];
InputDataFolder = [sourceDataPath, inputdir];
GTfolders = dir(GTFolder); % list the scenes
Inputfolders = dir(InputDataFolder); % list the scenes

GTfolders(1:2) = [];
Inputfolders(1:2) = [];
sceneNum = length(GTfolders);
idx = 0;
for iScene = 1 : sceneNum
    idx_s = 0;
    GTsceneName = GTfolders(iScene).name;
    InputsceneName = Inputfolders(iScene).name;
    GTsceneName(end-3:end) = [];
    fprintf('Generating training data of Scene_%s in Dataset %s......\t\t', GTsceneName);
    GTPath = [GTFolder, GTfolders(iScene).name];
    InputPath = [InputDataFolder, Inputfolders(iScene).name];
    data_GT = load(GTPath);
    data_In = load(InputPath);

    LF_GT = data_GT.LF; %读取光场数据  
    LF_In = data_In.LF; %读取光场数据  
    [U, V, H, W, ~] = size(LF_GT);

    for h = 1 : stride : H-patchsize+1
        for w = 1 : stride : W-patchsize+1                
            InSAI = single(zeros(U, V, patchsize, patchsize, 3));%single类型占4个字节
            GTSAI = single(zeros(U, V, patchsize, patchsize, 3));
            for u = 1 : U
                for v = 1 : V
                    SAItemp_GT = squeeze(LF_GT(u, v, h:h+patchsize-1, w:w+patchsize-1, :));
                    SAItemp_In = squeeze(LF_In(u, v, h:h+patchsize-1, w:w+patchsize-1, :));

                    GTSAI(u, v, :, :, :) = SAItemp_GT;
                    InSAI(u, v, :, :, :) = SAItemp_In;                       
                end
            end
            idx = idx + 1;
            SavePath_H5 = [SavePath, num2str(idx,'%05d'),'.h5'];
            h5create(SavePath_H5, '/data', size(InSAI), 'Datatype', 'single');
            h5write(SavePath_H5, '/data', single(InSAI), [1,1,1,1,1], size(InSAI));
            h5create(SavePath_H5, '/label', size(GTSAI), 'Datatype', 'single');
            h5write(SavePath_H5, '/label', single(GTSAI), [1,1,1,1,1], size(GTSAI));                
        end
    end
    fprintf([num2str(idx), ' training samples have been generated\n']);
end


