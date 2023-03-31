classdef Fun05_GetRANInfo < handle
    properties (SetAccess=private)
        v_allCell;
        n_allCell;
        n_allCircle;
        v_allSector;
        v_gridRSRP;
        m_allRSRP;
        v_RSRPSpace;
        v_TrafficSpace;
        v_actSpace;
        v_actSpaceProb;
        v_RC_stateSpace;
        
        v_Parameter;
        m_traficPoson;
        v_atvCell0;
        v_atvGrid0;
        
        v_atvCell;
        v_atvGrid;
        v_atvCell_RSRP;
        v_atvCell_ITF;
        v_atvCell_SINR;
        v_atvCell_TGO;
    end
    
    methods
        function obj=Fun05_GetRANInfo(v_Parameter)
            obj.v_Parameter=v_Parameter;
            obj.v_atvCell0=[];
            obj.v_atvGrid0=[];            
        end
        function v_flName=readRANInfo(obj)
            funName=mfilename;      FIdx=find('F'==funName);        funNameIdx=funName([FIdx+3:FIdx+4]);
            str_Path=pwd;           tIdx=find('t'==str_Path);       pIdx=find('p'==str_Path);
            fL_LimitSINR=dir(fullfile(str_Path,'CSV*_LmtSINR_Stp*.csv'));
            n_fL_LimitSINR=size(fL_LimitSINR,1);
            if n_fL_LimitSINR==0
                Fun04_CalLimitSINR();
                fL_LimitSINR=dir(fullfile(str_Path,'CSV*_LmtSINR_Stp*.csv'));
                n_fL_LimitSINR=size(fL_LimitSINR,1);
            end
            SIdx=find('S'==fL_LimitSINR.name);
            m_LimitSINR=csvread(fL_LimitSINR.name);
            
            obj.v_allCell=m_LimitSINR(:,1);
            obj.n_allCell=max(obj.v_allCell);
            obj.n_allCircle=max(m_LimitSINR(:,2));
            obj.v_allSector=m_LimitSINR(:,3);
            obj.v_gridRSRP=m_LimitSINR(:,4);
            obj.m_allRSRP=m_LimitSINR(:,9:end);
            v_flName.selfName=['CSV' funNameIdx '_DQN' 'StatData' '_' fL_LimitSINR.name([SIdx(3):end])];
        end
        function setRANInfoSpace(obj, v_idxMemory, memory_buffer)
            obj.v_TrafficSpace=obj.v_Parameter(3):obj.v_Parameter(3):obj.v_Parameter(1)*1.1;
            obj.v_RSRPSpace=min(obj.v_gridRSRP)-obj.v_Parameter(4):obj.v_Parameter(3):max(obj.v_gridRSRP)+obj.v_Parameter(4);            
            obj.v_actSpace=0:obj.v_Parameter(3):obj.v_Parameter(4);
            n_actSpace=length(obj.v_actSpace);           
            v_action(min(v_idxMemory(1),v_idxMemory(2)))=0;
            for iM=1:min(v_idxMemory(1),v_idxMemory(2))
                v_action(iM)=memory_buffer(iM).action;
            end            
            m_actSpaceProb=tabulate(v_action);
            v_actSpaceProb=m_actSpaceProb(:,3);
            n_actSpaceProb=length(v_actSpaceProb);
            if n_actSpace==n_actSpaceProb
                obj.v_actSpaceProb=v_actSpaceProb'/100;
            else                
                obj.v_actSpaceProb(1:n_actSpace)=1/n_actSpace;
            end
            obj.v_RC_stateSpace=[length(obj.v_TrafficSpace) length(obj.v_RSRPSpace)];            
        end
        function setPreTrafic(obj,m_traficPoson, v_atvCell0,v_atvGrid0)
            if isempty(m_traficPoson)
                obj.m_traficPoson=zeros(2, obj.n_allCell);
                obj.m_traficPoson(1,:)=random('poisson',obj.v_Parameter(1),1,obj.n_allCell);
                obj.m_traficPoson(1,:)=(obj.m_traficPoson(1,:)<=obj.v_Parameter(1)).*obj.m_traficPoson(1,:);
                obj.m_traficPoson(2,:)=random('poisson',obj.v_Parameter(2),1,obj.n_allCell);                
            else
                obj.m_traficPoson=m_traficPoson;                
            end
            obj.standardTraficPoson();
            obj.v_atvCell0=v_atvCell0;
            obj.v_atvGrid0=v_atvGrid0;
        end
        function decTTIofTraficPoson(obj)
            while ~sum(obj.m_traficPoson(1,:))
                obj.m_traficPoson(1,:)=random('poisson',obj.v_Parameter(1),1,obj.n_allCell);
                obj.m_traficPoson(1,:)=(obj.m_traficPoson(1,:)<=obj.v_Parameter(1)).*obj.m_traficPoson(1,:);
                obj.m_traficPoson(2,:)=random('poisson',obj.v_Parameter(2),1,obj.n_allCell);                
            end
            while sum(find(0>=obj.m_traficPoson(2,:)))<1
                obj.m_traficPoson(2,:)=obj.m_traficPoson(2,:)-1;
                obj.m_traficPoson=(obj.m_traficPoson>0).*obj.m_traficPoson;
            end
            obj.standardTraficPoson();
        end
        function updateAtvCellUser(obj)
            obj.v_atvCell=find(0==obj.m_traficPoson(2,:));
            vt_atvCell=find(0<obj.m_traficPoson(1,obj.v_atvCell));
            obj.v_atvCell=obj.v_atvCell(vt_atvCell);
            obj.v_atvCell=sort(obj.v_atvCell);
            n_atvCell=length(obj.v_atvCell);
            obj.v_atvGrid=zeros(1,n_atvCell);
            vt_delIdx=zeros(n_atvCell);
            for iAC=1:n_atvCell
                i_atvCell=obj.v_atvCell(iAC);
                if ismember(i_atvCell, obj.v_atvCell0)
                    idx=find(i_atvCell==obj.v_atvCell0);
                    obj.v_atvGrid(iAC)=obj.v_atvGrid0(idx);
                else
                    v_i_atvCell=find(i_atvCell==obj.v_allCell);
                    n_i_atvCell=length(v_i_atvCell);
                    if n_i_atvCell
                        i_atvCell_Grid=unidrnd(n_i_atvCell);
                        i_atvCell_Grid=v_i_atvCell(i_atvCell_Grid);
                        obj.v_atvGrid(iAC)=i_atvCell_Grid;
                    else
                        vt_delIdx(iAC)=1;                        
                    end
                end
            end
            if sum(vt_delIdx)
                vt_delIdx=find(1==vt_delIdx);
                obj.v_atvCell(vt_delIdx)=[];
                clear vt_delIdx;
            end
            obj.v_atvCell_RSRP=obj.v_gridRSRP(obj.v_atvGrid)';
        end
        function calPerformanceTrafic(obj,v_atvLPRSRP)
            obj.decTTIofTraficPoson();
            v_natvCell=1:obj.n_allCell;     v_natvCell(obj.v_atvCell)=[];   n_natvCell=length(v_natvCell);
            mt_allRSRP=obj.m_allRSRP(obj.v_atvGrid,:);
            v_RCAllRSRP=size(mt_allRSRP);
            mt_allRSRP(:,v_natvCell)=-995;
            if (isempty(v_atvLPRSRP) & (obj.n_allCell-n_natvCell))
                v_atvLPRSRP(obj.n_allCell-n_natvCell)=0;                
            else
                for iatvCell=1:length(obj.v_atvCell)
                    mt_allRSRP(:,obj.v_atvCell(iatvCell))=mt_allRSRP(:,obj.v_atvCell(iatvCell))-v_atvLPRSRP((iatvCell));
                end
            end            
            v_atvCell_RSRP=obj.v_atvCell_RSRP-v_atvLPRSRP;
            obj.v_atvCell_ITF=(10*log10(sum(10.^(mt_allRSRP/10),2)+10^(-125/10)))';
            obj.v_atvCell_SINR=v_atvCell_RSRP-obj.v_atvCell_ITF;
            obj.v_atvCell_TGO=10*log2(1+10.^(obj.v_atvCell_SINR/10));
        end
        
        function decTraficPoson(obj)            
            obj.m_traficPoson(1,obj.v_atvCell)=obj.m_traficPoson(1,obj.v_atvCell)-obj.v_atvCell_TGO;
            obj.m_traficPoson(2,:)=obj.m_traficPoson(2,:)-1;
            obj.m_traficPoson=(obj.m_traficPoson>0).*obj.m_traficPoson;            
                    
            vt_traficCell=find(0<obj.m_traficPoson(1,:));            
            v_natvCell=1:obj.n_allCell;     v_natvCell(vt_traficCell)=[];            n_natvCell=length(v_natvCell);
            rnd_trafic=rand;
            if rnd_trafic>=((length(obj.v_atvCell)/obj.n_allCell)^(1/.99))
                nt_atvCell=ceil(rnd_trafic*n_natvCell);
                vt_atvCell=sort(randsample(v_natvCell,nt_atvCell));
                obj.m_traficPoson(1,vt_atvCell)=random('poisson',obj.v_Parameter(1),1,nt_atvCell);
                obj.m_traficPoson(1,:)=(obj.m_traficPoson(1,:)<=obj.v_Parameter(1)).*obj.m_traficPoson(1,:);
                obj.m_traficPoson(2,vt_atvCell)=random('poisson',obj.v_Parameter(2),1,nt_atvCell);                
            end
            obj.standardTraficPoson();
        end
        
        function standardTraficPoson(obj)
            vt_0_idx=find(0==obj.m_traficPoson(1,:));
            obj.m_traficPoson(:,vt_0_idx)=0;
        end
    end
end