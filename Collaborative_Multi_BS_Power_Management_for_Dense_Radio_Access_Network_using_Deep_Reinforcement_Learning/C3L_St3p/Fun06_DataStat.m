function Fun06_DataStat()

clc;
close all;
% delete *fig *jpg;
str_Path=pwd;
tIdx=find('t'==str_Path);
pIdx=find('p'==str_Path);

fL_URET=dir(fullfile(str_Path,['CSV05_*_Stp*.csv']));
n_fL_URET=size(fL_URET,1);
if n_fL_URET~=8
    v_Input=[2 100; 2 200; 5 100; 5 200];
    for hIdx=1:4
        ;
        Fun05_DQN(hIdx,v_Input(hIdx,:))
    end
    fL_URET=dir(fullfile(str_Path,['CSV05_*_Stp*.csv']));
    n_fL_URET=size(fL_URET,1);
end

fL_RANInfo=dir(fullfile(str_Path,['DQN_RANInfo*.mat']));
lineIdx=find('_'==fL_URET(1).name);
BIdx=find('B'==fL_URET(1).name);
n_allCell=str2num(fL_URET(1).name(lineIdx(end)+1:BIdx(end)-1));
m_aveSuccess=[];             m_aveEpisode=[];         m_aveReward=[];          m_aveRtTGOLos=[];
m_aveRtEgySv=[];             m_aveLPRSRP=[];          m_aveITF=[];             m_aveSleepReward=[];   
m_aveTGO=[];                 m_aveLPTGO=[];           m_avePower=[];           m_aveLPPower=[];
m_aveCumSuccess=[];             m_aveCumEpisode=[];         m_aveCumReward=[];          m_aveCumRtTGOLos=[];
m_aveCumRtEgySv=[];             m_aveCumLPRSRP=[];          m_aveCumITF=[];             m_aveCumSleepReward=[];   
m_aveCumTGO=[];                 m_aveCumLPTGO=[];           m_aveCumPower=[];           m_aveCumLPPower=[];
str_Tmnl=[];
v_rURET=[];
for iNFL=1:n_fL_URET
    m_URET=csvread(fL_URET(iNFL).name);
    [r_URET c_URET]=size(m_URET);
    if ~isempty(v_rURET) && r_URET~=v_rURET(end)
        continue;
    end
    t_idxT=find('t'==fL_URET(iNFL).name);
    v_idxOverline=find('_'==fL_URET(iNFL).name);
    str_Tmnl=[str_Tmnl fL_URET(iNFL).name(7) fL_URET(iNFL).name([t_idxT(3)+2:v_idxOverline(2)-1])];
    
    m_aveSuccess=[m_aveSuccess m_URET(:,2)];
    m_aveEpisode=[m_aveEpisode m_URET(:,3)];
    m_aveReward=[m_aveReward m_URET(:,4)];
    m_aveRtTGOLos=[m_aveRtTGOLos m_URET(:,5)];
    m_aveRtEgySv=[m_aveRtEgySv m_URET(:,6)];    
    m_aveLPRSRP=[m_aveLPRSRP m_URET(:,20)];
    m_aveITF=[m_aveITF m_URET(:,15)];
    m_aveSleepReward=[m_aveSleepReward m_URET(:,7)];
    m_aveTGO=[m_aveTGO m_URET(:,8)];
    m_aveLPTGO=[m_aveLPTGO m_URET(:,8)-m_URET(:,9)];
    m_avePower=[m_avePower m_URET(:,16)];
    m_aveLPPower=[m_aveLPPower m_URET(:,17)];
    
    m_aveCumSuccess=[m_aveCumSuccess m_URET(:,19)];
    m_aveCumEpisode=[m_aveCumEpisode m_URET(:,20)];
    m_aveCumReward=[m_aveCumReward m_URET(:,21)];
    m_aveCumRtTGOLos=[m_aveCumRtTGOLos m_URET(:,22)];
    m_aveCumRtEgySv=[m_aveCumRtEgySv m_URET(:,23)];    
    m_aveCumLPRSRP=[m_aveCumLPRSRP m_URET(:,29)];
    m_aveCumITF=[m_aveCumITF m_URET(:,32)];
    m_aveCumSleepReward=[m_aveCumSleepReward m_URET(:,24)];
    m_aveCumTGO=[m_aveCumTGO m_URET(:,25)];
    m_aveCumLPTGO=[m_aveCumLPTGO m_URET(:,25)-m_URET(:,26)];
    m_aveCumPower=[m_aveCumPower m_URET(:,33)];
    m_aveCumLPPower=[m_aveCumLPPower m_URET(:,34)];
    
    if iNFL==1
        [val idx]=min(m_URET(:,5));
        load(fL_RANInfo(1).name);
        atvCellState=struct_RANInfo(idx).atvCellState;
        atvCellStateLP=struct_RANInfo(idx).atvCellStateLP;
        m_RSRPITFTGO=[atvCellState(1,:) ;atvCellState([3 4 6],:);atvCellStateLP([3 4 6],:);atvCellState([3 4 6],:)-atvCellStateLP([3 4 6],:)];
    end
    v_rURET=[v_rURET r_URET];
end

m_aveStatData=[mean(m_aveSuccess); mean(m_aveEpisode); mean(m_aveReward); mean(m_aveRtTGOLos);...
    mean(m_aveRtEgySv); mean(m_aveLPRSRP); mean(m_aveITF); mean(m_aveSleepReward)];
m_stdStatData=[std(m_aveSuccess); std(m_aveEpisode); std(m_aveReward); std(m_aveRtTGOLos);...
    std(m_aveRtEgySv); std(m_aveLPRSRP); std(m_aveITF); mean(m_aveSleepReward)];
csvwrite(['CSV06_aveStatData.csv'], [m_aveStatData; -3*ones(1,size(m_stdStatData,2)); m_stdStatData]);

valLineWidth=3;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Reward for \eta and \beta 
h1=figure(1);
set(h1, 'outerposition', get(0,'screensize'));
subplot(1,2,1);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
bar([m_aveStatData(3,[1 3 2 4])]')
set(gca,'XTick', 1:4, 'XTickLabel',{'(100,2dBW)','(200,2dBW)','(100,5dBW)','(200,5dBW)'})
hleg=legend('Proposed','Location','NorthWest');
set(hleg,'FontName','Times New Roman','FontSize',20,'FontWeight','normal')
title('(1) Overall average consumption factor','FontName','Times New Roman','Color','k','FontSize',20); 
set(gca,'FontSize',20,'Fontname', 'Times New Roman');
xlabel('Diverse values of (N, \DeltaP_{max})');    ylabel('Consumption factor (in Mbps/dBW)');
set(gca,'position',[0.08 0.13 0.4 0.80])
subplot(1,2,2);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(m_aveCumReward(:,[1]),'--r','linewidth',valLineWidth)
plot(m_aveCumReward(:,[2]),'--m','linewidth',valLineWidth)
hleg=legend('Proposed: N=100,\DeltaP_{max}=2dBW','Proposed: N=100,\DeltaP_{max}=5dBW','Location','SouthEast');
set(hleg,'FontName','Times New Roman','FontSize',20,'FontWeight','normal')
title('(2) Cumulative average consumption factor','FontName','Times New Roman','Color','k','FontSize',20); 
set(gca,'FontSize',20,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Consumption factor (in Mbps/dBW)');
set(gca,'position',[0.58 0.13 0.4 0.80])
saveas(h1,['Pic01_Reward_0.fig'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Transmit power and throughput variation
h2=figure(2);
set(h2, 'outerposition', get(0,'screensize'));
subplot(2,2,1);
set(gca,'position',[0.06 0.59 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(1) Average throughput of BSs for (N=100, \DeltaP_{max}=2dBW)','FontName','Times New Roman','Color','k','FontSize',16);
plot(m_aveCumLPTGO(:,[1]),'--b','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Value (in Mbps)');
set(gca,'FontSize',16,'Fontname', 'Times New Roman');

subplot(2,2,2);
set(gca,'position',[0.54 0.59 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(2) Average downlink power of BSs for (N=100, \DeltaP_{max}=2dBW)','FontName','Times New Roman','Color','k','FontSize',16);
plot(m_aveCumLPPower(:,[1]),'--b','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Value (in dBW)');
set(gca,'FontSize',16,'Fontname', 'Times New Roman');

set(h2, 'outerposition', get(0,'screensize'));
subplot(2,2,3);
set(gca,'position',[0.06 0.095 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(3) Avergae throughput of BSs for (N=100, \DeltaP_{max}=5dBW)','FontName','Times New Roman','Color','k','FontSize',16);
plot(m_aveCumLPTGO(:,[2]),'--b','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Value (in Mbps)');
set(gca,'FontSize',16,'Fontname', 'Times New Roman');

subplot(2,2,4);
set(gca,'position',[0.54 0.095 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(4) Average downlink power of BSs for (N=100, \DeltaP_{max}=5dBW)','FontName','Times New Roman','Color','k','FontSize',16);
plot(m_aveCumLPPower(:,[2]),'--b','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Value (in dBW)');
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
saveas(h2,['Pic02_TgoPower_0.fig'])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Energy saving and throughput variation
h3=figure(3);
set(h3, 'outerposition', get(0,'screensize'));
subplot(2,3,1);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(m_aveCumLPRSRP(:,1), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(1) RSRP decline over sleep';'(N=100, \DeltaP_{max}=2dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)');
set(gca,'position',[0.0475 0.585 0.25 0.325])
subplot(2,3,4);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(m_aveCumLPRSRP(:,2), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(4) RSRP decline over sleep';'(N=100, \DeltaP_{max}=5dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)'); 
set(gca,'position',[0.0475 0.0825 0.25 0.325])

subplot(2,3,2);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(m_aveCumITF(:,1), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(2) Interference decline over sleep';'(N=100, \DeltaP_{max}=2dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)'); 
set(gca,'position',[0.365 0.585 0.25 0.325])
subplot(2,3,5);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(m_aveCumITF(:,2), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(5) Interference decline over sleep';'(N=100, \DeltaP_{max}=5dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)'); 
set(gca,'position',[0.365 0.0825 0.25 0.325])

subplot(2,3,3);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(-m_aveCumLPRSRP(:,1)+m_aveCumITF(:,1), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(3) [Interference Decline]-[RSRP Decline]';'(N=100,\DeltaP_{max}=2dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)')
set(gca,'position',[0.7 0.585 0.25 0.325])
subplot(2,3,6);
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
plot(-m_aveCumLPRSRP(:,2)+m_aveCumITF(:,2), '-k','linewidth',valLineWidth)
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
title({'(6) [Interference Decline]-[RSRP Decline]';'(N=100,\DeltaP_{max}=5dBW)'},'FontName','Times New Roman','Color','k','FontSize',16); 
set(gca,'FontSize',16,'Fontname', 'Times New Roman');
xlabel('Episode (in 1)');    ylabel('Value (in dBW)') 
set(gca,'position',[0.7 0.0825 0.25 0.325])
saveas(h3,['Pic03_RSRPITF_0.fig'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computational complexity
h4=figure(4);
set(h4, 'outerposition', get(0,'screensize'));
subplot(2,2,1);
set(gca,'position',[0.06 0.59 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(1) Overall average success ratio','FontName','Times New Roman','Color','k','FontSize',18); 
bar([m_aveStatData(1,[1 2 3 4])]')
set(gca,'XTick', 1:4, 'XTickLabel',{'(100, 2dBW)','(100, 5dBW)','(200, 2dBW)','(200, 5dBW)'})
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Diverse values of (N, \DeltaP_{max})');    ylabel('Success ratio (in 1)');
set(gca,'FontSize',18,'Fontname', 'Times New Roman');
subplot(2,2,2);
set(gca,'position',[0.56 0.59 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(2) Cumulative average success ratio','FontName','Times New Roman','Color','k','FontSize',18); 
plot(m_aveCumSuccess(:,[1]),'--k','linewidth',valLineWidth)
plot(m_aveCumSuccess(:,[2]),'--b','linewidth',valLineWidth)
hleg=legend('Proposed: N=100,\DeltaP_{max}=2dBW','Proposed: N=100,\DeltaP_{max}=5dBW','Best');
set(hleg,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Success ratio (in 1)');
set(gca,'FontSize',18,'Fontname', 'Times New Roman');

subplot(2,2,3);
set(gca,'position',[0.06 0.1 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(3) Overall average number of iterations','FontName','Times New Roman','Color','k','FontSize',18); 
bar([m_aveStatData(2,[1 2 3 4])]')
set(gca,'XTick', 1:4, 'XTickLabel',{'(100, 2dBW)','(100, 5dBW)','(200, 2dBW)','(200, 5dBW)'},...
    'YTick',0:50:200, 'YTickLabel',{'0','50','100','150','200'})
hleg=legend('Proposed','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',16,'FontWeight','normal')
xlabel('Diverse values of (N, \DeltaP_{max})');    ylabel('Iteration (in 1)');
set(gca,'FontSize',18,'Fontname', 'Times New Roman');

subplot(2,2,4);
set(gca,'position',[0.56 0.1 0.42 0.35])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
title('(4) Cumulative average number of iterations','FontName','Times New Roman','Color','k','FontSize',18);
plot(m_aveCumEpisode(:,[1]),'-b','linewidth',valLineWidth)
plot(m_aveCumEpisode(:,[3]),'-.b','linewidth',valLineWidth)
hleg=legend('Proposed:N=100,\DeltaP_{max}=2dBW','Proposed:N=200,\DeltaP_{max}=5dBW','Location','North');
set(hleg,'FontName','Times New Roman','FontSize',15,'FontWeight','normal')
xlabel('Episode (in 1)');    ylabel('Iteration (in 1)');
set(gca,'FontSize',18,'Fontname', 'Times New Roman');

saveas(h4,['Pic04_SuccessIteration_0.fig'])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Computational complexity
h5=figure(5);
set(h5, 'outerposition', get(0,'screensize'));
set(gca,'position',[0.075 0.16 0.88 0.75])
hold on;
grid on;    set(gca,'GridLineStyle',':','GridColor','k','GridAlpha',0.5)
box on;     set(gca,'LineWidth',1.5)
bar([m_RSRPITFTGO([8:10],:)]', 1)
hleg=legend('RSRP decline (in dBW)','Interference decline (in dBW)','Downlink data rate decline (in Mbps)','Location','Best');
set(hleg,'FontName','Times New Roman','FontSize',33,'FontWeight','normal')
xlabel('Active BS sequence (in 1)');    ylabel('Value');
set(gca,'FontSize',33,'Fontname', 'Times New Roman');
saveas(h5,['Pic05_RSRPITFTGO_1.fig'])
v_aveDeltaRSRPITF=10*log10(sum(10.^(m_RSRPITFTGO([2 3 5 6],:)/10),2)/n_allCell);
v_aveDeltaRSRPITF=[v_aveDeltaRSRPITF;v_aveDeltaRSRPITF([1 2])-v_aveDeltaRSRPITF([3 4]); sum(m_RSRPITFTGO(10,:))/n_allCell];


end
