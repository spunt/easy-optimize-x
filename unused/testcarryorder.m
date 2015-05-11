clear n
ncond = 2:10;
order = 2; 
count = 0; 
for i = 1:length(ncond)
    for ii = 1:length(order)
        seq = carryoverCounterbalance(ncond(i), order(ii), 1); 
        count = count + 1; 
        n(count) = length(seq); 
    end
end
disp(n)