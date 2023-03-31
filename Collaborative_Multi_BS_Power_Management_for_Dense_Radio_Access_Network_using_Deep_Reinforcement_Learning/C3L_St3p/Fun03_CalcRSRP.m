function Fun03_CalcRSRP()

%clear;
% clc;
global P_max;   %=15.2;
P_max=15.2;
funName=mfilename;
FIdx=find('F'==funName);
funNameIdx=funName([FIdx+3:FIdx+4]);

tic;
str_Path=pwd;
tIdx=find('t'==str_Path);
tIdx=tIdx(end);
pIdx=find('p'==str_Path);
stp=str2num(str_Path([tIdx+1:pIdx(1)-1]))

h_BS=25;        h_UT=1.5;
Radius=200;
str_Path=pwd;
f_AntAttenuate=dir(fullfile(str_Path, 'CSV*_AntAtu*.csv'));
n_f_AntAttenuate=size(f_AntAttenuate,1);
if n_f_AntAttenuate==0
    Fun02_GridBSRSRP();
    f_AntAttenuate=dir(fullfile(str_Path, 'CSV*_AntAtu*.csv'));
    n_f_AntAttenuate=size(f_AntAttenuate,1)
end
iCalcRSRP=3

m_AntAttenuate=csvread(f_AntAttenuate.name);
f_DisGrid2BS=dir(fullfile(str_Path, 'CSV*_DisGrdBS*.csv'));
m_DisGrid2BS=csvread(f_DisGrid2BS.name);
[r_DisGrid2BS c_DisGrid2BS]=size(m_DisGrid2BS);

c=3e8; f_c=1.8e9;
h_BP=4*h_BS*h_UT*f_c/c;
m_RSRP=zeros(r_DisGrid2BS,c_DisGrid2BS);
for iR=1:r_DisGrid2BS
    for jC=1:c_DisGrid2BS        
        d_2D=m_DisGrid2BS(iR,jC);
        s_attu=m_AntAttenuate(iR,jC);
        d_3D=sqrt(d_2D.^2+(h_BS-h_UT)^2);

        los_Pathloss_1=28+22*log10(d_3D)+20*log10(f_c/10^9);
        los_Pathloss_2=28+40*log10(d_3D)+20*log10(f_c/10^9)-9*log10(h_BP^2+(h_BS-h_UT)^2);
        if d_2D<=h_BP
            los_Pathloss=los_Pathloss_1;
        else
            los_Pathloss=los_Pathloss_2;
        end

        nlos_Pathloss=32.4+30*log10(d_3D)+20*log10(f_c/10^9)+0;

        ret_RSRP=P_max+17.15-1-13-nlos_Pathloss-s_attu+rand*10;
        m_RSRP(iR,jC)=ret_RSRP;
    end
end
csvwrite(['CSV' funNameIdx '_GridRSRP_Stp' num2str(stp) '_' num2str(length(m_RSRP(:,1))) 'Grid_' num2str(length(m_RSRP(1,:))) 'BS.csv'],m_RSRP)

end