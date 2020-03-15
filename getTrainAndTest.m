function [train test] = getTrainAndTest(trainSet, testSet)
    [totalClass, section, view_num] = size(trainSet);
    
    for view= 1 : view_num
        test{view,1}=[];
        for i = 1:totalClass
            test{view,1} = [test{view,1}; testSet{i,1,view}];
        end
    end

    for view = 1:view_num
        for i = 1:totalClass
            col_train{view,1} = [];
            for j = 1:section
                col_train{view} = [col_train{view}; trainSet{i, j, view}];
            end
            train{view, i} = col_train{view,1}(:, 1:end-1);
        end
    end
end
