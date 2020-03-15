% H = [1 0; 0 1];
% f = [-2; -6];
% A = [1 -1];
% b = 2;
% 
% x = quadprog(H,f,A,b,[],[],[])
A = 

%[x,~,~,~,~] = quadprog(H,f,A,b,[],[],[]);
%[x, fval,exitflag,output,lambda] = quadprog(H,f,A,b,[],[],[])
% x = quadprog(H,f,A,b,[],[],[]);
% A = [1 2 3; 4 5 6];
% C = 2 * eye(3);
% B = A * C * A';
% 
%   load heart_scale.mat  %加载测试数据集
%   svm_model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -t 0 -h 1');
%   %w = get_w(svm_model, Z', y);
%   %function w = get_w(svm_model, Z, y)
%     % Z : sample size * dim in here.
%     
%     w = zeros (size(heart_scale_inst, 2), 1);
%     sv_Z = heart_scale_inst(svm_model.sv_indices, : );
%     
%     sv_y = heart_scale_label(svm_model.sv_indices);
%     
%     for i =1 : svm_model.totalSV
%         %w = w + svm_model.sv_coef(i) * sv_y(i) * sv_Z ( i , : )';
%         w = w + svm_model.sv_coef(i) * sv_Z ( i , : )';
%     end
%     Y = sign(heart_scale_inst * w - svm_model.rho);
%     acc = Y == heart_scale_label;
%     Acc = find (acc ==1 );
% %end
% 
% %model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -t 0 -h 1');
% [predict_label, accuracy, dec_values] = svmpredict(heart_scale_label, heart_scale_inst, svm_model);  %用模型预测
% acc_t = predict_label==heart_scale_label;
% Acc_t = find (acc_t ==1 );


% A = magic(5); % 产生5阶魔方矩阵
% B = A(:) % 按列排，变成向量
% C = reshape(B,5,5);

