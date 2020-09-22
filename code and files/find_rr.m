
mat = dir('af-classification\training2017\*.mat'); 
length(mat);
for q = 1:length(mat) 
   name = strsplit(mat(q).name, '.');
   name = name{1};
   freq = 300;
%cont = load(mat(q).name); 
    [qrs_amp_raw,qrs_i_raw,delay] = qrs(name, freq, 0);

    %if size(qrs_amp_raw, 2) < 10
       %[qrs_amp_raw,qrs_i_raw,delay] = qrs(name, freq, 1);
       %f = 1
       %break; 
    %end
    
   %break;
   q
end