function Fun02_GridBSRSRP()

clear;
% clc;
funName=mfilename;
FIdx=find('F'==funName);
funNameIdx=funName([FIdx+3:FIdx+4]);

tic;
h_BS=25;        h_UT=1.5;
str_Path=pwd;
CIdx=find('C'==str_Path);
LIdx=find('L'==str_Path);
n_Circle=str2num(str_Path([CIdx+1:LIdx-1]));
tIdx=find('t'==str_Path);
tIdx=tIdx(end);
pIdx=find('p'==str_Path);
stp=str2num(str_Path([tIdx+1:pIdx(1)-1]));
fL_BSCdt=dir(fullfile(str_Path, 'CSV*_Cdt_*.csv'));
n_fL_BSCdt=size(fL_BSCdt,1);
if n_fL_BSCdt==0
    Fun01_NetCtnCdt();
    fL_BSCdt=dir(fullfile(str_Path, 'CSV*_Cdt_*.csv'));
    n_fL_BSCdt=size(fL_BSCdt,1);
end
iGridBSRSRP=2

fL_AntPattern=dir(fullfile(str_Path, 'CSV*_AntPtn_*.csv'));
n_fL_AntPattern=size(fL_AntPattern,1);
if n_fL_AntPattern==0
    Fun00_GenerateAntenna()
    fL_AntPattern=dir(fullfile(str_Path, 'CSV*_AntPtn_*.csv'));
    n_fL_AntPattern=size(fL_AntPattern,1);
end

for iFL=1:n_fL_BSCdt    
    m_BSCtnCdt=csvread(fL_BSCdt(iFL).name);
    m_AntPatten=40-csvread(fL_AntPattern(iFL).name);
    m_BSCdt=m_BSCtnCdt(:,[5 6]);
    m_AreaBorder=round([min(m_BSCdt)-200; max(m_BSCdt)+200]);
    m_GridCdt=[];
    for iX=m_AreaBorder(1,1):stp:m_AreaBorder(2,1)
        for jY=m_AreaBorder(1,2):stp:m_AreaBorder(2,2)
            m_GridCdt=[m_GridCdt; [iX jY]];
        end
    end
    [r_GridCdt c_GridCdt]=size(m_GridCdt);
    [r_BSCdt c_BSCdt]=size(m_BSCdt);
    m_DisGrid2BS=zeros(r_GridCdt,r_BSCdt);
    m_virtlBSCdt=m_BSCdt;
    m_virtlBSCdt(:,2)=m_virtlBSCdt(:,2)+50;
    m_H_Azimuth=zeros(r_GridCdt,r_BSCdt);
    m_V_Azimuth=zeros(r_GridCdt,r_BSCdt);
    m_AntAttenuate=zeros(r_GridCdt,r_BSCdt);
    for iG=1:r_GridCdt
        for iBS=1:r_BSCdt
            m_DisGrid2BS(iG,iBS)=sqrt((m_GridCdt(iG,1)-m_BSCdt(iBS,1))^2+(m_GridCdt(iG,2)-m_BSCdt(iBS,2))^2);
            
            t_virtlDisGrid2BS=sqrt((m_GridCdt(iG,1)-m_virtlBSCdt(iBS,1))^2+(m_GridCdt(iG,2)-m_virtlBSCdt(iBS,2))^2);
            t_H_azimuth=round(180*acos((50^2+m_DisGrid2BS(iG,iBS)^2-t_virtlDisGrid2BS^2)/(2*50*m_DisGrid2BS(iG,iBS)))/pi);
            if m_GridCdt(iG,1)<m_BSCdt(iBS,1)
                t_H_azimuth=t_H_azimuth+180;
            end
            t_H_azimuth=mod(t_H_azimuth,360);
            m_H_Azimuth(iG,iBS)=t_H_azimuth;
            
            t_V_azimuth=round(180*atan(m_DisGrid2BS(iG,iBS)/(h_BS-h_UT))/pi);
            t_V_azimuth=mod(180+t_V_azimuth,180);
            m_V_Azimuth(iG,iBS)=t_V_azimuth;
            
            t_H_azimuth=mod(t_H_azimuth,120);
            if t_H_azimuth>=60
                t_H_azimuth=t_H_azimuth-60;
            end
            m_AntAttenuate(iG,iBS)=m_AntPatten(t_H_azimuth+1,t_V_azimuth+1);
        end
    end
    csvwrite(['CSV' funNameIdx '_DisGrdBS_Stp' num2str(stp) '_' num2str(n_Circle) 'CL_' num2str(max(m_BSCtnCdt(:,1))) 'BS.csv'],m_DisGrid2BS)
    csvwrite(['CSV' funNameIdx '_H_AmGrdBS_Stp' num2str(stp) '_' num2str(n_Circle) 'CL_' num2str(max(m_BSCtnCdt(:,1))) 'BS.csv'],m_H_Azimuth)
    csvwrite(['CSV' funNameIdx '_V_AmGrdBS_Stp' num2str(stp) '_' num2str(n_Circle) 'CL_' num2str(max(m_BSCtnCdt(:,1))) 'BS.csv'],m_V_Azimuth)
    csvwrite(['CSV' funNameIdx '_AntAtuGrdBS_Stp' num2str(stp) '_' num2str(n_Circle) 'CL_' num2str(max(m_BSCtnCdt(:,1))) 'BS.csv'],m_AntAttenuate)
end

end