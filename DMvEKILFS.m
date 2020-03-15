function [w1, P] = DMvEKILFS( classOne , classTwo , C1, C2, inputInf)
%function [w1, P] = DMvEKILFS( classOne , classTwo , C1, C2, C3, inputInf)
    len_one = size(classOne{1}, 1);
    len_two = size(classTwo{1}, 1);
    y_1 = ones(len_one , 1); 
    y_2 = -1 * ones(len_two , 1);
    y = [y_1; y_2];
    num_sample = len_one + len_two;
    %Parameter Initialized
    P = cell(1, inputInf.view);
    X = cell(1, inputInf.view);
    for view = 1 : inputInf.view
       org_dim(view) = size(classOne{view}, 2);
       P{view} = rand(inputInf.dim, org_dim(view)); 
       X{view} = [classOne{view}; classTwo{view}]'; % org_dim * sample_size
    end
    
    Z = rand(inputInf.dim, num_sample);
    %Z = zeros(inputInf.dim, num_sample);
    w = rand(inputInf.dim, 1);
    w0 = 0.5;
    %h = rand(1, num_sample);
    
    %Optimization
    count = 0;
    while(1)
        count = count + 1;   
        %update projected sample Z
            for iter = 1 : num_sample
                    h(iter) = hinge_loss(y(iter), w, w0, Z( : , iter)); 
                    Z( : , iter) = update_zi( P, w, w0, X, y, h, iter, num_sample, inputInf, C2);
                    %Z( : , iter) = update_zi_test( P, w, w0, X, y, h, iter, num_sample, inputInf, C2);
            end
        fprintf('.......  Parameter Z has been updated.  .......\n') ;
        %update projection matrix P

        for view = 1 : inputInf.view
            P{view} = update_P(P{view}, Z, X{view}, inputInf, C1);
%             vec_P{view} = update_P(P{view}, Z, X{view}, inputInf, C1);
%             P{view} = reshape(vec_P{view}, inputInf.dim ,org_dim(view));
        end
        fprintf('.......  Parameter P has been updated.  .......\n') ;
        
        %update w and w0
        %svm_model = svmtrain(y, Z', '-c 10 -g 0.07');
        svm_model = svmtrain(y, Z', '-t 0 -c 1 ');
        w = get_w(svm_model, Z');
        w0 = -svm_model.rho;
        fprintf('.......  Parameter w && w0 has been updated.  .......\n') ;

        % Objective function converge or not.
        J1 = compute_J1(Z, P, X, num_sample, inputInf);
        J2 = C1 * compute_J2(P, inputInf);
        J3 = compute_J3(Z);
        J4 = 0.5 * norm(w, 2)^2;
        J5 = compute_J5(y, w, w0, Z);
        
        J(count) = J1 + J2 + J3 + J4 + J5;
        if count == 1
            obj_J(count) = J(count);
        else
             obj_J(count) = abs(J(count) -  J(count - 1));
        end
        
        if obj_J(count)  < inputInf.termination || count > inputInf.sizeIter
            break;
        end
    end
    w1 = [w',w0];
end

function h_i = hinge_loss(y_i, w, w0, z_i)
    h_i = max(0, 1 - y_i * (w' * z_i + w0));
end

function z_i = update_zi( P, w, w0, X, y, h, iter, num_sample, inputInf, C2)
    
    H = ((2 / (num_sample * inputInf.view)) + 2 * C2) * eye(inputInf.dim); 
    c = zeros(inputInf.dim, 1);
    for i = 1 : inputInf.view
        c = c - (2 / (num_sample * inputInf.view)) * (P{i} * X{i}( : , iter ));
    end
    A = -y(iter) * w';
    b = h(iter) - 1 + y(iter) * w0;
    [z_i, ~, exitflag, ~,~] = quadprog(H, c, A, b, [], [], []);
    
end
% function z_i = update_zi_test( P, w, w0, X, y, h, iter, num_sample, inputInf, C2)
%     temp_H = zeros (inputInf.dim, inputInf.dim);
%     temp_c = zeros (1, inputInf.dim);
%     for i = 1 : inputInf.view
%         temp_H = temp + P{i}'* P{i};
%         temp_c = temp_c + (-2 / (num_sample * inputInf.view))* P{i}' * X{view}(:, iter); 
%         %c = c + (-2 / (num_sample * inputInf.view)) * (P{i} * X{i}( : , iter ));
%     end
%     H = 2 / (num_sample * inputInf.view) * temp + (2 * C2 * eye(inputInf.dim));
%     A = -y(iter) * w';
%     b = h(iter) - 1 + y(iter) * w0;
%     z_i = quadprog(H, temp_c, A, b, [], [], []);
% 
% end
function P_v = update_P(P_v, Z, X_v, inputInf, C1)
    A = construct_D(P_v); 
    B = zeros(size(X_v, 1), size(X_v, 1));
    C = zeros(inputInf.dim ,size(X_v, 1));
    for i = 1 : size(X_v, 2)
        B = B + X_v( : , i) * X_v( : , i)';
        C = C + Z( : , i) * X_v( : , i)';
    end
%     vec_C = C( : );
%     vec_P = inv(kron(eye(size(X_v, 1)), C1 * A) + kron(1/(size(X_v, 2) * inputInf.view) * B, eye(inputInf.dim))) * vec_C;
%     P = reshape(vec_P, inputInf.dim ,size(X_v, 1));
    P_v = lyap(A,B,C);
end

function J1 = compute_J1(Z, P, X, num_sample, inputInf)
    J1 = 0 ;
    for i = 1 : num_sample
        for view = 1 : inputInf.view
            J1 = J1 + 1/(num_sample * inputInf.view) * norm(Z( : , i) - P{view} * X{view}( : , i), 2);
        end
    end
end

function J2 = compute_J2(P, inputInf)
    J2 = 0;
    for view = 1 : inputInf.view
        J2 = J2 + compute_L21_norm(P{view});
    end   
end

function J3 = compute_J3(Z)
    J3 = 0;
    for i = 1 : size(Z, 2)
        J3 = J3 + norm(Z( : , i))^2;
    end
end

function J5 = compute_J5(y, w, w0, Z)
    J5 = 0;
    for i = 1 : size(Z, 2)
        J5 = J5 + hinge_loss(y(i), w, w0, Z( : , i));
    end
end

function L21_norm = compute_L21_norm(P_v)
    L21_norm = 0;
    for i = size(P_v,1)
        L21_norm = L21_norm + norm(P_v(i, : ));
    end
end

function D = construct_D(P_v)
        D = zeros(size(P_v, 1), size(P_v, 1));
        for i = 1 : size(P_v, 1)
            D(i,i) = 1 / (2 * norm(P_v( i , : ), 2));
        end
end

function w = get_w(svm_model, Z)
    % Z : sample size * dim in here.    
    w = zeros (size(Z, 2), 1);
    sv_Z = Z(svm_model.sv_indices, : );   
    for i =1 : svm_model.totalSV
        w = w + svm_model.sv_coef(i) * sv_Z ( i , : )';
    end
end