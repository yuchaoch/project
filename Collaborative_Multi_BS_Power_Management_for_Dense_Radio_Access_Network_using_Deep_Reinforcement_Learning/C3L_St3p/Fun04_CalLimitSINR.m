function Fun04_CalLimitSINR()

%clear;
funName=mfilename;
FIdx=find('F'==funName);
funNameIdx=funName([FIdx+3:FIdx+4]);

close all;
delete CSV*_LimitSINR_Stp*.csv
str_Path=pwd;
fL_GridRSRP=dir(fullfile(str_Path,'CSV*_GridRSRP_Stp*.csv'));
n_fL_GridRSRP=size(fL_GridRSRP,1);
if n_fL_GridRSRP==0
    Fun03_CalcRSRP();
    fL_GridRSRP=dir(fullfile(str_Path,'CSV*_GridRSRP_Stp*.csv'));
    n_fL_GridRSRP=size(fL_GridRSRP,1);
end
CalLimitSINR=4

m_GridRSRP0=csvread(fL_GridRSRP.name);
[r_GridRSRP c_GridRSRP]=size(m_GridRSRP0);
[v_hRSRP v_hIdx]=max(m_GridRSRP0');
m_GridRSRP=m_GridRSRP0;
for iRGR=1:r_GridRSRP
    m_GridRSRP(iRGR,v_hIdx(iRGR))=-999;
end
fL_Cdt=dir(fullfile(str_Path,'CSV*_Cdt_Stp*.csv'));
m_BSCtnCdt=csvread(fL_Cdt.name);
m_LimitSINR=[v_hIdx' m_BSCtnCdt(v_hIdx,[2 4]) v_hRSRP' v_hRSRP'+125];
v_GridITF=10*log10(sum(10.^(m_GridRSRP/10),2)+10^(-125/10));
m_LimitSINR=[m_LimitSINR v_hRSRP'-v_GridITF v_GridITF zeros(r_GridRSRP,1) m_GridRSRP];
SIdx=find('S'==fL_Cdt.name);
csvwrite(['CSV' funNameIdx '_LmtSINR_' fL_Cdt.name([SIdx(2):end])], m_LimitSINR);


end
