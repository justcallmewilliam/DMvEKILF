function [emp_train , emp_Test , t_train] = kernel_mapping(train_class_one , train_class_two, test_data , kernelPerType , kernelPar)%���ӳ����ѵ���������Լ����Լ���ʱemp_train����emp_classOne����emp_classTwo
    %
    % train_data = {train_class_one , train_class_two } ;
    % test_data = [X_in_input_space ] ;
    %
    
    %train_class_one = train_data.classOne ;%��һ���ѵ����Ϊ40*4
    %train_class_two = train_data.classTwo ;%�ڶ����
    len_ClassOne = size(train_class_one , 1) ;%��һ����������൱������
    len_ClassTwo = size(train_class_two , 1) ;%�ڶ����
    
    [emp_trn_all , emp_Test , t_train] = emp_Generator([train_class_one ; train_class_two] , test_data , kernelPerType , kernelPar) ;%����ӳ����ѵ�������Լ�

    emp_classOne = emp_trn_all(1:len_ClassOne , :) ;%ѵ����ӳ������������һ��ģ�����Ҫ�ֿ�
    emp_classTwo = emp_trn_all(len_ClassOne+1 : len_ClassOne+len_ClassTwo , :);
    emp_train.emp_classOne = emp_classOne ;
    emp_train.emp_classTwo = emp_classTwo ;
    
    clear temp_emp emp_classOne emp_classTwo ;
end

function [emp_train , emp_Test , t_train] = emp_Generator(trainData , testData , kType , kPar)%ѵ�����Ͳ��Լ�һ��ӳ�䣬emp_train , emp_Testӳ����������t_train����ʱ��
    % start clock for trainData
    tic%����ʱ��tic ��toc�ѳ���Χ��������
    
    implicitKernel = Kernel(trainData , trainData , kType , kPar) ;%ѵ����ӳ��ĺ�,�˴�trainDataΪ[train_class_one ; train_class_two]
    [pc , variances , explained] = pcacov(implicitKernel);%[PC, LATENT, EXPLAINED] = pcacov(X) ��pc,latent,explained��������,����ֵ,����

    i = 1 ;
    label = 0 ;
    while variances(i) >= 1e-3 ;%����ֵ���ڵ���10^(-3)��latent,explained�����ɸߵ����ĺ�˳��ġ�
        if i+1 > size(variances,1) ;%�����������ֵ�ĸ���
            label = 1 ;
            break ;
        end;
        i = i + 1 ;    
    end;

    if label == 0 ;
        i = i - 1 ;
    end;

    index = 1 : i ;
    P = pc(: , index) ;%��������ֵ��������������
    R = diag(variances(index)) ;%X = diag(v,k)������v��Ԫ����Ϊ����X�ĵ�k���Խ���Ԫ�أ���k=0ʱ��vΪX�����Խ��ߣ���k>0ʱ��vΪ�Ϸ���k���Խ��ߣ���k<0ʱ��vΪ�·���k���Խ��ߡ�
    emp_train = implicitKernel * P * R^(-1/2) ;%�൱�������е�B=KAQ.^(-1/2)    
    t_train = toc ;
    
    kerTestMat = Kernel(testData ,trainData , kType , kPar) ;%�˻��Ĳ��Ծ�����Ҫ�����Լ�����ѵ����һ�����
    emp_Test = kerTestMat * P * R^(-1/2) ;  
end

