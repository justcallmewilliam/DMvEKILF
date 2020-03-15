function w1 = MultiKMHKS(classOne , classTwo , C1, C2, inputInf) 
    len_one = size(classOne{1}, 1);
    len_two = size(classTwo{1}, 1);
    IR = len_two/len_one;
    D = diag([IR*ones(1, len_one), ones(1, len_two)]);
    len_all = len_one + len_two;
    one_all = ones(len_all, 1);
    for i = 1: inputInf.M
        Y{i} = [[classOne{i}, ones(size(classOne{i}, 1), 1)]; -1*[classTwo{i}, ones(size(classTwo{i}, 1), 1)]];
        dim = size(Y{i}, 2);
        w0{i} = 0.5*ones(dim, 1);
        b0{i} = ones(len_all, 1)*inputInf.B(i);
        I = eye(dim);
        I(end, end) = 0;
        P{i} = pinv((1 + C2*(inputInf.M - 1)/inputInf.M)*Y{i}'*D*Y{i} + C1*I);
    end
    [L0, mean_out, b1] = getL(w0, b0, Y, one_all, D, inputInf, C1, C2);
    b0 = b1;
    iter = 1;
    while iter <= inputInf.sizeIter
        iter = iter + 1;
        for i = 1:inputInf.M
            w1{i} = P{i}*Y{i}'*D*(b0{i} + one_all + C2*(mean_out - Y{i}*w0{i})/inputInf.M);
        end
        w0 = w1;
        [L1, mean_out, b1] = getL(w0, b0, Y, one_all, D, inputInf, C1, C2);
        if (L1 - L0)'*(L1 - L0) <= inputInf.termination 
            break;
        end
        L0 = L1;
        b0 = b1;
    end
end

function [L, mean_out, b1] = getL(w, b, Y, one_all, D, inputInf, C1, C2)
    left = 0;
    mean_out = 0;
    for i = 1:inputInf.M
        temp = (Y{i}*w{i} - one_all - b{i})'*D*(Y{i}*w{i} - one_all - b{i}) - C1*w{i}'*w{i};
        left = left + temp;
        mean_out = mean_out + Y{i}*w{i};
        e{i} = Y{i}*w{i} - b{i} - one_all;
        b1{i} = b{i} + 0.99*(e{i} + abs(e{i}));
    end
    mean_out = mean_out/inputInf.M;
    right = 0;
    for i = 1:inputInf.M
        temp = C2*(Y{i}*w{i} - mean_out)'*(Y{i}*w{i} - mean_out);
        left = left + temp;
    end
    L = left + right;
end