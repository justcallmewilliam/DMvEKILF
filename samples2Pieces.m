function Segment = samples2Pieces(dataSet , segmentNum, totalClass) 
    %
    %dataSet is a structrue of {Class_1 , Class_2 , Class_3 , ...}
    %
    %totalClass = size(dataSet , 2) ;%样本类别数
    view_num = size(dataSet, 1);
    Segment = [] ;
    for i = 1 : totalClass
        classData = dataSet( : ,i) ;%第i类样本
        len = size(classData{1,1} , 1) ;
        index = randperm(len) ;%ep randperm(7)会出现 1 7 6 4 3 5 2的序列
        segSize = floor(len/segmentNum) ;%floor向左取整
        for k = 1 : segmentNum - 1            
            for view = 1 : view_num
                Segment{i, k, view} = classData{view, 1}(index(segSize * (k-1) + 1 : segSize * k) , :);  
            end
        end
        
        for view = 1 : view_num
            Segment{i,k+1,view} = classData{view,1}(index(segSize*(k) + 1 : len) , :) ;
        end
        
    end
%     Segment = Segment' ;
end