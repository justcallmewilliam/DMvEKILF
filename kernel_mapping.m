function [emp_train , emp_Test , t_train] = kernel_mapping(train_class_one , train_class_two, test_data , kernelPerType , kernelPar)%获得映射后的训练集，测试集，以及用时emp_train包含emp_classOne，和emp_classTwo
    %
    % train_data = {train_class_one , train_class_two } ;
    % test_data = [X_in_input_space ] ;
    %
    
    %train_class_one = train_data.classOne ;%第一类的训练集为40*4
    %train_class_two = train_data.classTwo ;%第二类的
    len_ClassOne = size(train_class_one , 1) ;%第一类的行数，相当于样本
    len_ClassTwo = size(train_class_two , 1) ;%第二类的
    
    [emp_trn_all , emp_Test , t_train] = emp_Generator([train_class_one ; train_class_two] , test_data , kernelPerType , kernelPar) ;%产生映射后的训练，测试集

    emp_classOne = emp_trn_all(1:len_ClassOne , :) ;%训练集映射后是两类合在一起的，所以要分开
    emp_classTwo = emp_trn_all(len_ClassOne+1 : len_ClassOne+len_ClassTwo , :);
    emp_train.emp_classOne = emp_classOne ;
    emp_train.emp_classTwo = emp_classTwo ;
    
    clear temp_emp emp_classOne emp_classTwo ;
end

function [emp_train , emp_Test , t_train] = emp_Generator(trainData , testData , kType , kPar)%训练集和测试集一起映射，emp_train , emp_Test映射后的样本，t_train所用时间
    % start clock for trainData
    tic%计算时间tic ，toc把程序围起来即可
    
    implicitKernel = Kernel(trainData , trainData , kType , kPar) ;%训练集映射的核,此处trainData为[train_class_one ; train_class_two]
    [pc , variances , explained] = pcacov(implicitKernel);%[PC, LATENT, EXPLAINED] = pcacov(X) 中pc,latent,explained特征向量,特征值,比重

    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 ;%特征值大于等于10^(-3)，latent,explained都是由高到低拍好顺序的。
        if i+1 > size(variances,1) ;%如果超过特征值的个数
            label = 1 ;
            break ;
        end;
        i = i + 1 ;    
    end;

    if label == 0 ;
        i = i - 1 ;
    end;

    index = 1 : i ;
    P = pc(: , index) ;%满足特征值条件的特征向量
    R = diag(variances(index)) ;%X = diag(v,k)以向量v的元素作为矩阵X的第k条对角线元素，当k=0时，v为X的主对角线；当k>0时，v为上方第k条对角线；当k<0时，v为下方第k条对角线。
    emp_train = implicitKernel * P * R^(-1/2) ;%相当于论文中的B=KAQ.^(-1/2)    
    t_train = toc ;
    
    kerTestMat = Kernel(testData ,trainData , kType , kPar) ;%核化的测试矩阵，需要将测试集放入训练集一起算的
    emp_Test = kerTestMat * P * R^(-1/2) ;  
end

