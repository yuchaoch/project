function Fun05_DQN(hIdx,v_Input)

close all;
tic;
para=v_Input(1);
n_maxItr=v_Input(2);
n_batch = 500; %3000;
n_Iterations = 20*n_batch;
isTrain=1;
alpha=0.1;
gamma=0.9;
v_Parameter=[100 10 .1 para];
hdl_RANInfo=Fun05_GetRANInfo(v_Parameter);
v_flName=hdl_RANInfo.readRANInfo();
v_flName.selfName=[v_flName.selfName([1:9]) num2str(v_Parameter(4)+n_maxItr) v_flName.selfName([10:end])];
memory_buffer(1) = struct('state',[1 2],'action',[0.1],'next_state',[1 2],'reward',[0.3],'done',[0]);
hdl_RANInfo.setRANInfoSpace([1 2],memory_buffer);
hdl_RANInfo.setPreTrafic([],[],[]);
hdl_RANInfo.decTTIofTraficPoson();
hdl_RANInfo.updateAtvCellUser();
hdl_RANInfo.calPerformanceTrafic([]);
hdl_SAE=Fun05_SAEnvironment(hdl_RANInfo);
v_hiddenLayer=[30 30];
hdl_DQN=Fun05_DQNEstimator(hdl_SAE,alpha,v_hiddenLayer);

if ~isTrain
    load(['DQN_weights' num2str(v_Parameter(4)+n_maxItr) 'dB' '.mat'],'-mat');
    hdl_DQN.setWeights(Weights);
end

n_memory = 50*n_batch;    %10000; %30000;
memory_buffer(1:n_memory) = struct('state',[],'action',[],'next_state',[],'reward',[],'done',[]);
i_memory = 0;
i_batch=1;
epsilon = 0.2;

struct_RANInfo(1).traficPoson=zeros(2, hdl_RANInfo.n_allCell);
struct_RANInfo(1).atvCellState=[];
struct_RANInfo(1).atvCellStateLP=[];
struct_RANInfo(1).v_actSpaceProb=[];

m_statData=zeros(n_batch,17);
v_Success(n_Iterations,1)=0;
h1=figure(hIdx);
set(h1, 'outerposition', get(0,'screensize'));
m_PlotData=zeros(2,1);
p = plot(m_PlotData(1,:),m_PlotData(2,:), 'EraseMode','background','MarkerSize',5);
grid on;
title(['Iteration' num2str(n_Iterations) 'maxReducedPower-' num2str(para+n_maxItr) 'dB']);
axis([0 100 0 1]);

clc;
n_mini_batch=n_batch/.2;
v_losVal(n_Iterations/n_mini_batch)=0;
i_losVal=0;
for itr=1:n_Iterations    
    done=0;
    state = hdl_DQN.SAEInfo.curState;
    while isempty(state)
        state = hdl_DQN.SAEInfo.getCurState();
    end
    struct_RANInfo(itr).traficPoson=hdl_DQN.SAEInfo.RANInfo.m_traficPoson;
    struct_RANInfo(itr).atvCellState=[hdl_DQN.SAEInfo.RANInfo.v_atvCell; hdl_DQN.SAEInfo.RANInfo.v_atvGrid; hdl_DQN.SAEInfo.RANInfo.v_atvCell_RSRP;...
        hdl_DQN.SAEInfo.RANInfo.v_atvCell_ITF; hdl_DQN.SAEInfo.RANInfo.v_atvCell_SINR; hdl_DQN.SAEInfo.RANInfo.v_atvCell_TGO];
    struct_RANInfo(itr).v_actSpaceProb=hdl_DQN.SAEInfo.RANInfo.v_actSpaceProb;   
    v_RCState=size(state);
    mt_statDataPerItr=zeros(n_maxItr,16);
    mt_Reward=zeros(n_maxItr,v_RCState(2));
    m3_ReplayMemory(1:n_maxItr) = struct('state',[],'action',[],'next_state',[],'reward',[],'done',[]);
    i_ActCnt=0;
    while ~done
        if i_ActCnt==n_maxItr            
            [val idx]=max(mt_statDataPerItr(:,3));
            m_statData(i_batch,:)=[v_RCState(2) mt_statDataPerItr(idx,:)];
            m_statData(i_batch,3)=idx;
            i_batch=i_batch+1;
            if isTrain
                replayMemory=m3_ReplayMemory(idx);
                replayMemory.done=mt_Reward(idx,:);
                memory_buffer=hdl_DQN.SAEInfo.updateReplayMemory(memory_buffer, n_memory, replayMemory, v_RCState(2));
                i_memory = i_memory + v_RCState(2);
                clear replayMemory m3_ReplayMemory;
            end 
            if 0==mod(itr,100)
                DQNLog=round([itr mt_statDataPerItr(idx, 1) idx v_RCState(2) mt_statDataPerItr(idx, [4 5])*100 v_Parameter(4) n_maxItr i_memory toc])
            end
            break;
        end
        i_ActCnt=i_ActCnt+1;
        
        [~,~, ~, m_outValue]=hdl_DQN.predict(state);        
        m_Policy=hdl_DQN.SAEInfo.EpsilonGreedyPolicy(epsilon, m_outValue);
        v_Act=zeros(1,v_RCState(2));       
        for iR=1:v_RCState(2)
            i_act = randsample(hdl_DQN.SAEInfo.RANInfo.v_actSpace,1,1,m_Policy(iR,:));
            v_Act(iR)=i_act;
        end
        [m_nextState m_EgySvTGOLosReward v_Done] = hdl_DQN.SAEInfo.step(v_Act);
        vt_atvCell_RSRPLP=hdl_DQN.SAEInfo.RANInfo.v_atvCell_RSRP-v_Act;
        struct_RANInfo(itr).atvCellStateLP=[hdl_DQN.SAEInfo.RANInfo.v_atvCell; hdl_DQN.SAEInfo.RANInfo.v_atvGrid; vt_atvCell_RSRPLP;...
                            hdl_DQN.SAEInfo.RANInfo.v_atvCell_ITF; hdl_DQN.SAEInfo.RANInfo.v_atvCell_SINR; hdl_DQN.SAEInfo.RANInfo.v_atvCell_TGO];
        v_CF=hdl_DQN.SAEInfo.RANInfo.v_atvCell_TGO./(15.2-v_Act);
        v_aveEgySvTGOLosReward=sum(m_EgySvTGOLosReward([1:5],:),2)/hdl_DQN.SAEInfo.RANInfo.n_allCell;
        v_aveEgySvTGOLosReward([1 3])=[v_aveEgySvTGOLosReward(5)/v_aveEgySvTGOLosReward(4) sum(v_CF)/hdl_DQN.SAEInfo.RANInfo.n_allCell];
        v_aveRSRPITF=(10*log10(sum(10.^(m_EgySvTGOLosReward([6:9],:)/10),2)/hdl_DQN.SAEInfo.RANInfo.n_allCell))';
        v_aveRSRPITF=[v_aveRSRPITF([1 2]) v_aveRSRPITF(1)-v_aveRSRPITF(2) v_aveRSRPITF([3 4]) v_aveRSRPITF(3)-v_aveRSRPITF(4) ...
            15.2*v_RCState(2)/hdl_DQN.SAEInfo.RANInfo.n_allCell (15.2*v_RCState(2)-sum(v_Act))/hdl_DQN.SAEInfo.RANInfo.n_allCell];
        m3_ReplayMemory(i_ActCnt).state=state;
        m3_ReplayMemory(i_ActCnt).action=v_Act;
        m3_ReplayMemory(i_ActCnt).next_state=m_nextState;
        m3_ReplayMemory(i_ActCnt).reward=v_CF;
        m3_ReplayMemory(i_ActCnt).done=v_Done;
        
        rwd=v_aveEgySvTGOLosReward(3);
        v_aveEgySvTGOLosReward(3)=sum(hdl_DQN.SAEInfo.RANInfo.v_atvCell_TGO./15.2)/hdl_DQN.SAEInfo.RANInfo.n_allCell;       
        
        mt_Reward(i_ActCnt,:)=v_CF;
        
        if v_aveEgySvTGOLosReward(5)<=0
            done=1;
            v_Success(itr)=1;
        end
        mt_statDataPerItr(i_ActCnt,:)=[done i_ActCnt rwd v_aveEgySvTGOLosReward' v_aveRSRPITF];
        mt_statDataPerItr(i_ActCnt,1)=max(done,mt_statDataPerItr(i_ActCnt,1));
        done=0;
    end
    
    if mod(itr,10)==0
        v_Itr=1:itr;
        set(p,'XData', v_Itr', 'YData',(cumsum(v_Success(1:itr)))./v_Itr');
        drawnow
        axis([0 ceil(itr/n_batch)*n_batch 0 1]);
    end
    
    if 0==mod(itr, 5*n_batch)
        mt_statData=[];
        if itr>5*n_batch
            mt_statData=csvread(v_flName.selfName);
        end
        mt_statData=[mt_statData; m_statData(:,:)];
        csvwrite(v_flName.selfName, mt_statData);
        clear mt_statData; 
        m_statData=zeros(n_batch,17);
        i_batch=1;
        save(['DQN_RANInfo' num2str(v_Parameter(4)+n_maxItr) 'dB' '.mat'],'struct_RANInfo');   
    end    
        
    if  (isTrain & (i_memory>n_batch) & (0==mod(itr,round(n_mini_batch))))     %isTrain
        vt_losVal(n_mini_batch)=0;
        hdl_DQN.SAEInfo.RANInfo.setRANInfoSpace([i_memory n_memory],memory_buffer);
        mini_batch = randsample(memory_buffer(1:min(i_memory,n_memory)),n_mini_batch);
        for iNB=1:n_mini_batch
            batch_state = mini_batch(iNB).state;
            batch_action = mini_batch(iNB).action;
            batch_next_state = mini_batch(iNB).next_state;
            batch_reward = mini_batch(iNB).reward;
            batch_done = mini_batch(iNB).done;
            [~, ~,batch_nextStateValues] = hdl_DQN.predict(batch_next_state);            
            if ~batch_done
                batch_target = batch_reward + gamma*max(batch_nextStateValues);
            else
                batch_target = batch_reward;
            end
            losVal=hdl_DQN.update(batch_state,batch_action,batch_target);
            vt_losVal(iNB)=losVal;
            if mod(iNB,100)==0
                trainLog=round([itr iNB n_mini_batch toc])
            end
        end
        i_losVal=i_losVal+1;        v_losVal(i_losVal)=mean(vt_losVal.^2)/2
        Weights = hdl_DQN.weights;
        save(['DQN_weights' num2str(v_Parameter(4)+n_maxItr) 'dB' '.mat'],'Weights');        
    end
end
m_statData=csvread(v_flName.selfName);
m_statData(:,2)=v_Success;
v_RCStatData=size(m_statData);
mt_statData=m_statData;
v_RStatData=(1:v_RCStatData(1))';
mt_statData(:,[1 2])=[-3*ones(v_RCStatData(1),1) cumsum(m_statData(:,2))./v_RStatData];
mt_statData(:,[3 4])=[cumsum(m_statData(:,3))./v_RStatData cumsum(m_statData(:,4))./v_RStatData];
mt_statData(:,[5 6])=[cumsum(m_statData(:,5))./v_RStatData cumsum(m_statData(:,6))./v_RStatData];
mt_statData(:,[7 8])=[cumsum(m_statData(:,7))./v_RStatData cumsum(m_statData(:,8))./v_RStatData ];
mt_statData(:,[9 10])=[cumsum(m_statData(:,9))./v_RStatData cumsum(m_statData(:,10))./v_RStatData];
mt_statData(:,[11 12])=[cumsum(m_statData(:,11))./v_RStatData cumsum(m_statData(:,12))./v_RStatData];
mt_statData(:,[13 14])=[cumsum(m_statData(:,13))./v_RStatData cumsum(m_statData(:,14))./v_RStatData];
mt_statData(:,[15 16])=[cumsum(m_statData(:,15))./v_RStatData cumsum(m_statData(:,16))./v_RStatData];
mt_statData(:,[17])=[cumsum(m_statData(:,17))./v_RStatData];
csvwrite(v_flName.selfName, [m_statData mt_statData]);
toc
clear mt_statData m_statData Weights struct_RANInfo memory_buffer v_Success;


end