function [Acc, t_train] = MultiKMHKS_Fuc(trainSet , testSet, C1, C2, inputInf, categories)
    [totalClass, view_num] = size(trainSet) ;
    
    lenTest = size(testSet{1,1}, 1) ;
    testLable = testSet{1,1}( :, size(testSet{1,1}, 2)) ;
    resultMat = zeros(lenTest , totalClass) ; %存放结果的矩阵为测试集长度，与类别数   
    t_train = 0 ;
    for i = 1 : totalClass
        for view = 1: view_num
            classOne{view, 1} = trainSet{i, view} ;
        end
        
        for j = i +1 : totalClass
            for view = 1: view_num
                classTwo{view, 1} = trainSet{j, view} ;
            end
            %classTwo = trainSet{j,:} ;
            tic;
            %fprintf('The current classing data is %s and %s\n' , categories{i} , categories{j});
            [class_one , class_two , testData] = GenerateEmpiricalData(classOne , classTwo , testSet, inputInf);
            %w = MultiKMHKS(class_one , class_two , C1, C2, inputInf) ;
            %[w, P] = DMvEKILFS( class_one , class_two , C1, C2, inputInf);
            [w, P] = PT_DMvEKILFS( class_one , class_two , C1, C2, inputInf);
            z_test_data = GenTestSampleToIntactSpace( testData, P, inputInf );
            
            t = toc;
            t_train = t_train + t;
            %temp = class4test(w, testData, inputInf, lenTest);
            temp = class4test(w, z_test_data, inputInf, lenTest);
            %resultMat是一个测试样本数为行，类别数为列的矩阵
            indexClassOne = find(temp == 1) ;%找到判定为第一类的位置
            resultMat(indexClassOne , i) = resultMat(indexClassOne , i) + 1 ;%判定为i类，则对应i列的样本+1
            indexClassTwo = find(temp == -1) ;
            resultMat(indexClassTwo , j) = resultMat(indexClassTwo , j) + 1 ;%判定为j类，则对应j列的样本+1
        end
    end    
    [C finalClass] = max((resultMat'));
    Acc = size(find(finalClass' == testLable),1)/lenTest;
end

% function temp = class4test(w, testData, inputInf, lenTest)
%     test_set = zeros(lenTest, 1);
%     for i = 1: inputInf.view
%         data = [testData{i}, ones(lenTest, 1)];
%         test_set = test_set + data*w{i};
%     end
%     temp = sign(test_set);
%     temp(find(temp == 0)) = 1;
% end

% intact space start
function temp = class4test(w, testData, inputInf, lenTest)
    test_set = zeros(lenTest, 1);
    
    data = [testData, ones(lenTest, 1)];
    test_set = test_set + data * w';
    
    temp = sign(test_set);
    temp(find(temp == 0)) = 1;
end
% intact space end

function [class_one , class_two , testData] = GenerateEmpiricalData(org_one , org_two , testSet , inputInf)
    M = inputInf.view ;  %M是视角数量
    %view_num = size (org_one, 1);
    class_one=cell(M , 1) ;
    class_two=cell(M , 1) ;
    testData = cell(M , 1) ;
    tempKPar = aveRBFPar(org_one , org_two) ;
    inputInf.kPar = inputInf.kdelta .* tempKPar' ;
    
    %trainData.classOne = org_one ;%org_one two都是40*4列的矩阵
    %trainData.classTwo = org_two ;
    t_train = 0 ;
    for i = 1 : M ;
        kernelType = char(inputInf.kType) ;%本例中为RBF核         
        [emp_train , emp_Test , t] = kernel_mapping(org_one{ i, 1}, org_two{ i, 1}, testSet{i, 1}(:, 1:end - 1), kernelType , inputInf.kPar(i)) ;
        t_train = t_train + t ;
        
        class_one(i) = {emp_train.emp_classOne} ;
        class_two(i)= {emp_train.emp_classTwo} ;
        testData(i) = {emp_Test} ;
    end
     %testData =  testData';
end


function par=aveRBFPar(data1 , data2)%得到RBF核的参数delta，
    
    for i = 1 : size(data1,1)
        
        data{i} = [data1{i}; data2{i}];
        data_size(i) = size(data{1,i}, 1);
        mat_temp{i} = sum(data{1,i}.^2, 2) * ones(1, data_size(i)) + ones(data_size(i), 1)*sum(data{1,i}.^2, 2)' - 2 * data{1,i} * data{1,i}';
        tempMean(i) = (1/data_size(i)^2) * sum(sum(mat_temp{1,i},1),2) ;
        par(i) = sqrt(tempMean(i)) ;
    end
    
end