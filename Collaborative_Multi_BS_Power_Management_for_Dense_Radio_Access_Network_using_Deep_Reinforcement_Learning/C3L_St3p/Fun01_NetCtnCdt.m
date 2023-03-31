function Fun01_NetCtnCdt()

clear;
clc;
% close all;
delete *Cdt*.csv;

funName=mfilename;
FIdx=find('F'==funName);
funNameIdx=funName([FIdx+3:FIdx+4]);

str_Path=pwd;
tIdx=find('t'==str_Path);
tIdx=tIdx(end);
pIdx=find('p'==str_Path);
stp=str2num(str_Path([tIdx+1:pIdx(1)-1]));
CIdx=find('C'==str_Path);
LIdx=find('L'==str_Path);
n_Circle=str2num(str_Path([CIdx+1:LIdx-1]));

fL_AntPattern=dir(fullfile(str_Path, 'CSV*_AntPtn_*.csv'));
n_fL_AntPattern=size(fL_AntPattern,1);
if n_fL_AntPattern==0
    Fun00_GenerateAntenna()
    fL_AntPattern=dir(fullfile(str_Path, 'CSV*_AntPtn_*.csv'));
    n_fL_AntPattern=size(fL_AntPattern,1);
end
iNetCtnCdt=1

Radius=200;     d_BS2BS=sqrt(3)*Radius/2;   n_Sector=6;
m_BSCtnCdt=[1 0 1 0 0];
index=1;
for iCL=1:n_Circle
    n_BSPerCL=n_Sector*iCL;
    
    m_BSIDPerCL=[];
    for jBPC=1:n_BSPerCL
        m_BSIDPerCL=[m_BSIDPerCL; [n_Sector*(iCL-1)+jBPC+1 iCL jBPC]];
    end
    m_BSXYPerCL=[];
    for jS=1:n_Sector
        m_BSXYPerSCT=[d_BS2BS*iCL*sin((jS-1)*pi/3) d_BS2BS*iCL*cos((jS-1)*pi/3)];
        for jCL=2:iCL
            if jS==1
                m_BSXYPerSCT=[m_BSXYPerSCT; [(jCL-1)*Radius*0.75 (iCL-0.5*(jCL-1))*d_BS2BS]];
            elseif jS==2
                m_BSXYPerSCT=[m_BSXYPerSCT; [m_BSXYPerSCT(1,1) m_BSXYPerSCT(1,2)-(jCL-1)*d_BS2BS]];
            elseif jS==3
                m_BSXYPerSCT=[m_BSXYPerSCT; [(jCL-1)*Radius*0.75 -(iCL-0.5*(jCL-1))*d_BS2BS]];
            elseif jS==4
                m_BSXYPerSCT=[m_BSXYPerSCT; [-(jCL-1)*Radius*0.75 -(iCL-0.5*(jCL-1))*d_BS2BS]];
            elseif jS==5
                m_BSXYPerSCT=[m_BSXYPerSCT; [m_BSXYPerSCT(1,1) m_BSXYPerSCT(1,2)+(jCL-1)*d_BS2BS]];
            elseif jS==6
                m_BSXYPerSCT=[m_BSXYPerSCT; [-(jCL-1)*Radius*0.75 (iCL-0.5*(jCL-1))*d_BS2BS]];
            end
        end
        m_BSXYPerCL=[m_BSXYPerCL; m_BSXYPerSCT];
    end
    m_BSCtnCdt=[m_BSCtnCdt; [m_BSIDPerCL m_BSXYPerCL]];
end
[r_BSCtnCdt c_BSCtnCdt]=size(m_BSCtnCdt);
% m_BSCtnCdt(:,[end-1 end])=m_BSCtnCdt(:,[end-1 end])+150*rand(r_BSCtnCdt,2);
m_BSCtnCdt(:,[end-1 end])=m_BSCtnCdt(:,[end-1 end]);
m_BSCtnCdtAzimuth=[];
t_Dis=3;
for iR=1:r_BSCtnCdt    
    m_BSCtnCdtAzimuth=[m_BSCtnCdtAzimuth; [m_BSCtnCdt(iR,[1:3]) 0 m_BSCtnCdt(iR,[4 5])+[0 t_Dis]]];
    m_BSCtnCdtAzimuth=[m_BSCtnCdtAzimuth; [m_BSCtnCdt(iR,[1:3]) 120 m_BSCtnCdt(iR,[4 5])+[sqrt(3)*t_Dis/2  -t_Dis/2]]];
    m_BSCtnCdtAzimuth=[m_BSCtnCdtAzimuth; [m_BSCtnCdt(iR,[1:3]) 240 m_BSCtnCdt(iR,[4 5])+[-sqrt(3)*t_Dis/2 -t_Dis/2]]];
end
m_BSCtnCdt=m_BSCtnCdtAzimuth;

[r_BSCtnCdt c_BSCtnCdt]=size(m_BSCtnCdt);
m_BSCtnCdt=[[1:r_BSCtnCdt]' m_BSCtnCdt(:,[2:end])];
csvwrite(['CSV' funNameIdx '_Cdt_Stp' num2str(stp) '_' num2str(n_Circle) 'CL_' num2str(max(m_BSCtnCdt(:,1))) 'BS.csv'], m_BSCtnCdt);

end

