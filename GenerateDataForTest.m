%% Initialization
clear all;
clc;
%close all;
addpath(genpath('./Functions/'))


%% Parameters setting

angRes = 5;
level = 100;
sourceDataPath = './Test_mat_5x5/';
sourceDatasets = dir(sourceDataPath);%获得指定文件夹下的所有子文件夹和文件
sourceDatasets(1:2) = [];
datasetsNum = length(sourceDatasets);

SavePath = ['./5x5Dataset/TestData_', num2str(level), '_', 'LLE', '_', num2str(angRes), 'x', num2str(angRes), '/'];
if exist(SavePath, 'dir')==0
    mkdir(SavePath);
end
inputdir = ['/1_', num2str(level), '/'];
GTFolder = [sourceDataPath, '/1/'];
InputDataFolder = [sourceDataPath, inputdir];
GTfolders = dir(GTFolder); % list the scenes
InputDataFolder = [sourceDataPath];
Inputfolders = dir(InputDataFolder); % list the scenes

GTfolders(1:2) = [];
Inputfolders(1:2) = [];
sceneNum = length(Inputfolders);
idx = 0;
for iScene = 1 : sceneNum
    idx_s = 0;
    GTsceneName = GTfolders(iScene).name;
    InputsceneName = Inputfolders(iScene).name;
    InputsceneName(end-3:end) = [];
    GTsceneName(end-3:end) = [];
    fprintf('Generating test data of Scene_%s......\n', GTsceneName);
    GTPath = [GTFolder, GTfolders(iScene).name];
    InputPath = [InputDataFolder, Inputfolders(iScene).name];
    data_GT = load(GTPath);
    data_In = load(InputPath);

    LF_GT = data_GT.LF; %读取光场数据  
    LF_In = data_In.LF; %读取光场数据  
    [U, V, H, W, ~] = size(LF_In);
    
    while mod(H, 2) ~= 0
        H = H - 1;
    end
    while mod(W, 2) ~= 0
        W = W - 1;
    end

    LF_GT = LF_GT(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
    LF_In = LF_In(0.5*(U-angRes+2):0.5*(U+angRes), 0.5*(V-angRes+2):0.5*(V+angRes), 1:H, 1:W, 1:3); % Extract central angRes*angRes views
          
    InSAI = single(zeros(angRes, angRes, H, W, 3));%single类型占4个字节
    GTSAI = single(zeros(angRes, angRes, H, W, 3));
    for u = 1 : U
        for v = 1 : V
            SAItemp_GT = squeeze(LF_GT(u, v, :, :, :));
            SAItemp_In = squeeze(LF_In(u, v, :, :, :));

            GTSAI(u, v, :, :, :) = SAItemp_GT;
            InSAI(u, v, :, :, :) = SAItemp_In;                       
        end
    end
    
    SavePath_H5 = [SavePath, '/', InputsceneName, '.h5'];
    h5create(SavePath_H5, '/data', size(InSAI), 'Datatype', 'single');
    h5write(SavePath_H5, '/data', single(InSAI), [1,1,1,1,1], size(InSAI));
    h5create(SavePath_H5, '/label', size(GTSAI), 'Datatype', 'single');
    h5write(SavePath_H5, '/label', single(GTSAI), [1,1,1,1,1], size(GTSAI));                

end


