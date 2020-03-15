clear all;
clc;
%dirctory = './DataSet/Caltech101-Datasets/' ;
ResultName = strcat('result' , '.txt');
saveResultName = strcat('.\' , ResultName);
%for roundId = 1 : 26
    clc;
    fileName = '7_100_Caltech101';
    load (fileName);    
    
    clear(fileName);
    
    postName = strcat(fileName , '.mat');
    saveMatName =strcat('.\report\' , postName); % '.\report\iris_v.mat' ;
    postName = strcat(fileName , '.txt');
    saveFileName = strcat('.\report\' , postName);
    
    warning off;
    fid=fopen(saveFileName,'w');
    fclose(fid);

    totalCycle = 5;
    totalClass = size(new_categories, 2) ;
    
    dataSet = samples2Pieces(lite_sample, totalCycle, totalClass);

    inputInf.C1 = [10^-2; 10^-1; 10^0; 10^1; 10^2];
    inputInf.C2 = [10^-2; 10^-1; 10^0; 10^1; 10^2];
    
    %inputInf.lamda = [10^-2; 10^-1; 10^0; 10^1; 10^2];
    
    %inputInf.view = 1 ;        % <= 3,核的数量
    inputInf.view = size(lite_sample, 1);         %视角数
    
    
    inputInf.C = 2^(-4) * ones(inputInf.view+1 , 1) ;      % C，相当于每个核的比重
    inputInf.kdelta = 2^(-3)+(2^3 - 2^(-3))*rand(inputInf.view,1) ;
                      %  [10^-2; 10^-1; 10^0; 10^1; 10^2]
    inputInf.kType = 'rbf' ;
    inputInf.C((inputInf.view+1),1) = 0.01 ; %2^(-0) ;             % lamda
    inputInf.R = 0.99 * ones(inputInf.view,1) ;%学习率
    inputInf.B = 1e-6 * ones(inputInf.view,1) ;%bl
    inputInf.sizeIter = 100 ;
    inputInf.termination = 1e-3 ;%迭代终止条件
    inputInf.dim = 20; % dimension of the learned subspace
    
    %inputInf.alpha = 2.3849 ;
    inputInf.alpha = 0.3869 ;%%0.3869 The best parameter so far

    %start the first cycle
    FinalRes = [];
    FinalRecord = [];
    for iterC1 = 2 : 2
    %for iterC1 = 1 : size(inputInf.C1)
        C1 = inputInf.C1(iterC1, :);
        fid=fopen(saveFileName,'a');
        fprintf('Current C1: %f \n' , C1);
        fprintf(fid,'Current C1: %f \n' , C1);
        fclose(fid);
        for iterC2 = 1 : 1
        %for iterC2 = 1 : size(inputInf.C2)
            C2 = inputInf.C2(iterC2);
            fid=fopen(saveFileName,'a');
            fprintf('Current C2: %f \n' , C2);
            fprintf(fid,'Current C2: %f \n' , C2);
            fclose(fid);
            %for iterC3 = 1 : size(inputInf.lamda)
%                 C3 = inputInf.lamda(iterC3)
%                 fid=fopen(saveFileName,'a');
%                 fprintf('Current C3: %f \n' , C3);
%                 fprintf(fid,'Current C3: %f \n' , C3);
%                 fclose(fid);
                res = [];
                for i = 1:totalCycle
                    %learn the best parameter
                    testSet = dataSet(:, mod(3+i, totalCycle) + 1, :);
                    trainSet = dataSet;
                    trainSet(:, mod(3+i, totalCycle) + 1, :) = [];
                    [train test] = getTrainAndTest(trainSet, testSet);
                    [Acc t_train] = MultiKMHKS_Fuc(train' , test, C1, C2, inputInf, new_categories);
                    res = [res; [Acc t_train]];
                    fid=fopen(saveFileName,'a');
                    fprintf('The %d cycle --- Acc: %f  --- time: %f \n' , i , Acc, t_train);
                    fprintf(fid,'The %d cycle --- Acc: %f  --- time: %f \n' , i , Acc, t_train);
                    fclose(fid);
                end
                res(totalCycle+1 , :) = mean(res) ;%10轮交叉验证，11行纪录其均值
                res(totalCycle+2 , :) = std(res(1:totalCycle , :)) ;%12行纪录其方差
                FinalRes = [FinalRes ; {res}] ;

                fid=fopen(saveFileName,'a');
                fprintf('.......  mean AUC = %f\tstd = %f ........\n' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
                fprintf(fid,'.......  mean AUC = %f\tstd = %f ........\n' , res(totalCycle+1 , 1) , res(totalCycle+2 , 1)) ;
                fprintf('.......  mean time = %f\tstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
                fprintf(fid,'....... mean time = %f\tstd = %f .......\n' , res(totalCycle+1 , 2) , res(totalCycle+2 , 2)) ;
                fclose(fid);

                FinalRecord = [FinalRecord; [res(totalCycle+1 , :) res(totalCycle+2 , :) C1 C2]];
            %end
            
        end   
    end
    [maxValue , maxIndex] = max(FinalRecord(:, 1)) ;%1 is Gmean
    maxRes = FinalRes{maxIndex} ;
    
    fid=fopen(saveFileName,'a');
    fprintf('----------------------------------------\n') ;
    fprintf(fid,'----------------------------------------\n') ;
    fprintf('.......  Acc = %f\tstd = %f ........\t\n' , maxRes(totalCycle+1 , 1) , maxRes(totalCycle+2 , 1)) ;
    fprintf(fid,'.......  Acc = %f\tstd = %f ........\t\n' , maxRes(totalCycle+1 , 1) , maxRes(totalCycle+2 , 1)) ;
    fprintf('.......  time = %f\ttstd = %f .......\n' , maxRes(totalCycle+1 , 2) , maxRes(totalCycle+2 , 2)) ;
    fprintf(fid,'.......  time = %f\ttstd = %f .......\n' , maxRes(totalCycle+1 , 2) , maxRes(totalCycle+2 , 2)) ;
    fclose(fid);
    
    fid=fopen(saveResultName,'a');
    fprintf(fid,' Acc = %f \t std = %f  \t time = %f \t tstd = %f \n' ,  maxRes(totalCycle+1 , 1), maxRes(totalCycle+2 , 1), maxRes(totalCycle+1 , 2) , maxRes(totalCycle+2 , 2)) ;
    fclose(fid);
    
    savedObj.FinalRes = FinalRes ;
    savedObj.FfinalRecord = FinalRecord ;
    save(saveMatName , 'savedObj') ;



