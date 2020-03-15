function testsample = GenTestSampleToIntactSpace( org_testsample, P, inputInf )
%Project test sample by the learned projection matrix to the intact space.
%org_testsample: view * sample_size * dim 
    testsample = zeros(inputInf.dim, size(org_testsample{1}, 1));
    for view = 1 : size(P, 2)
        testsample = testsample + 1/size(P, 2) * P{view}' * org_testsample{view}';
    end
    testsample = testsample';  %sample_size * dim
end

