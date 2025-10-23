#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 21:35:36 2022

@author: wlcc
"""

import numpy as np
import pandas as pd
import sys, time, openpyxl, os, math, random, gc, datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.ion()

########################################################################################################################
class RANINfo:
    __t_start=time.time()
    __curDir=sys._getframe().f_code.co_filename
    __step= 2
    __circ= 10
    __n_BS = 0
    __xlsxName='RANInfo_C'+str(__circ)+'L_St'+str(__step)+'p.xlsx'
    __xlsxNameR=r'RANInfo_C'+str(__circ)+'L_St'+str(__step)+'p.xlsx'
    __h_BS = 25
    __h_UT=1.5
    __P_max=15.2
    __docHeader=None
    __v_THRScenario=[3.36, 38.23, 0.78, 1.82, 2.54, 2.15, 2.46]
    __n_DF = 10
    __pd_Q_table = None
    __l_index = None
    __v_actions = None
    __log = "log.txt"
    __log_file = None
    __log_txt = ""
    __pd0_atvCell = None
    __pd0_atv_GridRSRP = None
    __all_CellID = []
    __l_index_Q_table = []
    __pd_log_Q_table = None
    __i_step = 0
    __pd_4LmtSINR = None
    __i_file = 0
    __i_itr = 0
    __l_step_itr = []
    __l_pd_rawQ_table = None
    __l_pd_static_table = None
    __n_step = 200
    __pd_static_actions = None
    __pd_static_RSRP = None
    __pd_static_ITF_dec = None
    __pd_static_SINR_dec = None
    __pd_static_THR_dec = None
    __pd_static_RSRP_oft = None
    __pd_static_Satify_dec = None
    __pd_static_Reward = None
    __n_itr = 100
    __i_step_dur = 10
    __v_reward_weight = 0

########################################################################################################################
    def __init__(self, circ, step, n_step, n_itr, v_reward_weight, v_rt):
        self.__circ=circ
        self.__step=step
        self.__xlsxName='RANInfo_C'+str(self.__circ)+'L_St'+str(self.__step)+'p.xlsx'
        self.__xlsxNameR=r'RANInfo_C'+str(self.__circ)+'L_St'+str(self.__step)+'p.xlsx'
        self.__docHeader='OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/CSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p_'
        self.__v_cluster_RSRP_dec = np.linspace(0, 12, 5).astype(np.int8)
        self.__delt = int(self.__v_cluster_RSRP_dec[1]-self.__v_cluster_RSRP_dec[0])
        v_actions = np.linspace(0, self.__delt, self.__delt+1)
        self.__v_actions = np.sort(list(set(np.hstack((0, v_actions)))))
        self.__n_step = n_step
        self.__n_itr = n_itr
        self.__v_reward_weight = v_reward_weight
        self.__v_rt = v_rt
        
        if not os.path.exists('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p'):
            os.makedirs('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p')
        if not os.path.exists('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/net'):
            os.makedirs('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/net')
        # if os.path.exists('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log'):
        #     shutil.rmtree('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log')
        if os.path.exists('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log'):
            pass
        else:
            os.makedirs('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log/sta')
           
        self.__log_file = open(self.__docHeader + self.__log, mode='w')
        
        return

########################################################################################################################
    def var_name(var,all_var=locals()):
        return [var_name for var_name in all_var if all_var[var_name] is var][0]

########################################################################################################################
    def CDFHistPlot(self, fig, gs, v_gs, v_data, strTitle, v_xlim_min_max):
        ax = fig.add_subplot(gs[v_gs[0], v_gs[1]])
        plt.grid(linestyle="-.", color="k", linewidth='0.5')
        n, bins, patches = plt.hist(x=v_data, bins=self.__n_step*2, color='green', edgecolor = 'blue')
        self.cdfplot(v_data, strTitle, 'k-', np.max(n))
        if v_xlim_min_max[1]>=-50:
            plt.xlim([-0.5, 1.2*v_xlim_min_max[1]])            
        # plt.ylim([-.05*np.max(n), np.max(n)*1.05])
        plt.title(strTitle)
        plt.xlabel('Value');   plt.ylabel('CDF & Count')
        
        return

########################################################################################################################
    def selfSort(self, xdata):
        nums=xdata
        sorted_nums=sorted(enumerate(nums), key=lambda x:x[1])
        idx=[i[0] for i in sorted_nums]
        nums=[i[1] for i in sorted_nums]
        return np.array(nums), np.array(idx)

########################################################################################################################    
    def smooth(self, data, weight=0.9):
        last = data[0]  # First value in the plot (first timestep)
        smoothed = list()
        for point in data:
            smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

#################################################################################################################################### 
    def cdfplot(self, xdata, strLabel, strStyle, n_):
        x=np.sort(xdata)
        y=(1.*np.arange(len(xdata))/(len(xdata)-1))*n_
        plt.plot(x, y, strStyle, label=strLabel, linewidth=3.0)
        
        return

########################################################################################################################    
    def plot_rewards(self, rewards, tag='train'):
        sns.set()
        plt.figure()  # 创建一个图形实例，方便同时多画几个图
        plt.title("Reward")
        plt.xlabel('epsiodes')
        plt.plot(rewards, label='rewards')
        plt.plot(self.smooth(rewards), label='smoothed')
        plt.legend()
        plt.savefig(f"{tag}ing_curve.png")
        plt.show()

########################################################################################################################
    def write2Excel(self, strexcelFile, sheetName, pd_DF, indexFalse):
        # print('%s: [L%s]_[%s]_[%s]_[%s], %.1fs' % (self.__curDir[-10:], sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
        #                             sys._getframe(1).f_code.co_name, sheetName, (time.time()-self.__t_start)))
        strFile=Path(strexcelFile)        
        if  strFile.is_file():
            fileStat=os.stat(strFile)
            if fileStat.st_size>0:
                wb=openpyxl.load_workbook(strexcelFile)
                writer=pd.ExcelWriter(strexcelFile, mode='a', engine='openpyxl', if_sheet_exists="replace")                
            else:
                os.remove(strFile)
                wb=None
                writer=pd.ExcelWriter(strFile)
        else:
            wb=None
            writer=pd.ExcelWriter(strFile)
        
        if ((not wb is None) and (sheetName in wb.sheetnames)):
            ws=wb[sheetName]
            wb.remove(ws)
        pd_DF.to_excel(writer, sheet_name=sheetName, index=indexFalse)

        writer.close()

        return

########################################################################################################################
    def circshift(u,shiftnum1,shiftnum2):
        h,w = u.shape
        if shiftnum1 < 0:
            u = np.vstack((u[-shiftnum1:,:],u[:-shiftnum1,:]))
        else:
            u = np.vstack((u[(h-shiftnum1):,:],u[:(h-shiftnum1),:]))
        if shiftnum2 > 0:
            u = np.hstack((u[:, (w - shiftnum2):], u[:, :(w - shiftnum2)]))
        else:
            u = np.hstack((u[:,-shiftnum2:],u[:,:-shiftnum2]))
        return u

########################################################################################################################
    def GenAntena(self):
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start)))
        Radius = 200;        f = 3e10;        lamda = (3e8) / f;        beta = 2 * np.pi / lamda;
        maxGain=40;        n=8;                m=8;            v_RowCol=[360,181]
        t=np.linspace(0.0001,2*np.pi-.0001,v_RowCol[0])
        d=lamda/4
        W=beta*d*np.cos(t)
        z11=((n/2)*W)-n/2*beta*d
        z12=((1/2)*W)-1/2*beta*d
        F1=np.sin(z11)/(n*np.sin(z12))
        K1=abs(F1)*maxGain/max(abs(F1))
        t = np.linspace(0.0001,2*np.pi-.0001,v_RowCol[1])
        W=beta*d*np.cos(t)
        d=lamda/4
        z21=((m/2)*W)-m/2*beta*d
        z22=((1/2)*W)-1/2*beta*d
        F2=np.sin(z21)/(m*np.sin(z22))
        K2=abs(F2)
        K3 = np.transpose(K1)
        K0 = np.kron(K3,K2)
        v_VAntenna=K0[0:K2.shape[0]]
        minIdx = np.argmax(v_VAntenna)
        minIdx3dB = np.argmin(abs(v_VAntenna - 37))
        dgr_Antenna3dB=abs(minIdx-minIdx3dB)
        dgr_Down_Antenna=dgr_Antenna3dB+round(180*np.arctan(self.__h_BS/Radius)/np.pi)
        K0 = np.reshape(K0,(K1.shape[0],K2.shape[0]))
        K0 = np.roll(K0,dgr_Down_Antenna,axis=1)

        strIndex=['H'+str(i+1) for i in range(K0.shape[0])]
        strColum=['V'+str(i+1) for i in range(K0.shape[1])]
        pd_DF = pd.DataFrame(data=np.round(K0,3), index=strIndex, columns=strColum)
        del K0
        gc.collect()

        pd_DF.to_csv((self.__docHeader+'0GenAntena.csv').replace('p/CSV_','p/net/CSV_'))
        del pd_DF
        gc.collect()

        return

########################################################################################################################
    def NetCtnCnt(self):
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start)))

        Radius=200;                d_BS2BS=np.sqrt(3)*Radius/2;                n_Sector=6
        m_BSCtnCdt=[1,0,1,0,0]
        for iCL in range(1,self.__circ+1):
            n_BSPerCL=n_Sector*iCL

            m_BSIDPerCL=None
            for jBPC in range(1,n_BSPerCL+1):
                if m_BSIDPerCL is None:
                    m_BSIDPerCL=[n_Sector*(iCL-1)+jBPC+1, iCL, jBPC]
                else:
                    m_BSIDPerCL=np.vstack((m_BSIDPerCL,[n_Sector*(iCL-1)+jBPC+1, iCL, jBPC]))

            m_BSXYPerCL=None;
            for jS in range(1,n_Sector+1):
                v_BSXYPerSCT=[d_BS2BS*iCL*np.sin((jS-1)*np.pi/3),d_BS2BS*iCL*np.cos((jS-1)*np.pi/3)]
                m_BSXYPerSCT=v_BSXYPerSCT
                for jCL in range(2,iCL+1):
                    if jS==1:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[(jCL-1)*Radius*0.75,(iCL-0.5*(jCL-1))*d_BS2BS]))
                    elif jS==2:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[v_BSXYPerSCT[0],v_BSXYPerSCT[1]-(jCL-1)*d_BS2BS]))
                    elif jS==3:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[(jCL-1)*Radius*0.75,-(iCL-0.5*(jCL-1))*d_BS2BS]))
                    elif jS==4:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[-(jCL-1)*Radius*0.75,-(iCL-0.5*(jCL-1))*d_BS2BS]))
                    elif jS==5:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[v_BSXYPerSCT[0], v_BSXYPerSCT[1]+(jCL-1)*d_BS2BS]))
                    elif jS==6:
                        m_BSXYPerSCT=np.vstack((m_BSXYPerSCT,[-(jCL-1)*Radius*0.75, (iCL-0.5*(jCL-1))*d_BS2BS]))
                m_BSXYPerSCT=np.round(m_BSXYPerSCT, 3)

                if m_BSXYPerCL is None:
                    m_BSXYPerCL=m_BSXYPerSCT
                else:
                    m_BSXYPerCL=np.vstack((m_BSXYPerCL,m_BSXYPerSCT))

            m_BSCtnCdt=np.vstack((m_BSCtnCdt,np.hstack((m_BSIDPerCL,m_BSXYPerCL))))

        m_BSCtnCdtAzimuth=[0]*6
        t_Dis=3
        for iR in range(m_BSCtnCdt.shape[0]):
            m_BSCtnCdtAzimuth=np.vstack((m_BSCtnCdtAzimuth, np.hstack(([m_BSCtnCdt[iR,[0,1,2]], 0, m_BSCtnCdt[iR,[3,4]]+[0,t_Dis]]))))
            m_BSCtnCdtAzimuth=np.vstack((m_BSCtnCdtAzimuth, np.hstack(([m_BSCtnCdt[iR,[0,1,2]], 120, m_BSCtnCdt[iR,[3,4]]+[np.sqrt(3)*t_Dis/2,-t_Dis/2]]))))
            m_BSCtnCdtAzimuth=np.vstack((m_BSCtnCdtAzimuth, np.hstack(([m_BSCtnCdt[iR,[0,1,2]], 240, m_BSCtnCdt[iR,[3,4]]+[-np.sqrt(3)*t_Dis/2,-t_Dis/2]]))))

        m_BSCtnCdt=m_BSCtnCdtAzimuth[1:,:]
        m_BSCtnCdt=np.vstack((np.linspace(1, m_BSCtnCdt.shape[0],m_BSCtnCdt.shape[0]), np.round(m_BSCtnCdt[:,1:].T,3)))
        pd_DF=pd.DataFrame(data=np.round(m_BSCtnCdt.T,3), columns=['Cell ID', 'Circle_ID', 'BS ID for Circle','Azimuth', 'BS X', 'BS Y'])
        del m_BSCtnCdt
        gc.collect()

        pd_DF.to_csv((self.__docHeader+'1BSInfo.csv').replace('p/CSV_','p/net/CSV_'))
        del pd_DF
        gc.collect()
        
        return

########################################################################################################################
    def GridBSRSRP(self):
        
        Because my paper was plagiarized by someone with academic misconduct. So the code for this article is accessed with restriction. if you want the full code, please reach me with yuchaoch@126.com. 
        
        return

########################################################################################################################
    def CalcRSRP(self):
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start)))

        for iD in range(self.__n_DF):
            strTReadCsv = (self.__docHeader+'2AntAttuate'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_')
            try:
                pd_DF = pd.concat([pd_DF, pd.read_csv(strTReadCsv)])
            except:
                pd_DF = pd.read_csv(strTReadCsv)
        m_AntAttenuate=pd_DF.values[:,1:]
        del pd_DF
        gc.collect()        
        for iD in range(self.__n_DF):
            strTReadCsv = (self.__docHeader+'2DisGrid2BS'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_')
            try:
                pd_DF = pd.concat([pd_DF, pd.read_csv(strTReadCsv)])
            except:
                pd_DF = pd.read_csv(strTReadCsv)
        m_DisGrid2BS=pd_DF.values[:,1:]
        del pd_DF
        gc.collect()

        f_c=1.8e9;
        m_RSRP=np.zeros((m_DisGrid2BS.shape[0], m_DisGrid2BS.shape[1]))
        mm_RSRP = sys.getsizeof(m_RSRP)/1e6
        for iR in range(m_DisGrid2BS.shape[0]):
            if 0==(np.abs(iR)%10000) or 5001==(np.abs(iR)%10000) :
                print('%.0fs: %s [%s iR Prog] = [%.f/%.f, %.3f, %.2fMB]' % ((time.time()-self.__t_start), sys._getframe().f_lineno, sys._getframe(0).f_code.co_name, 
                                                                           iR/1000, m_DisGrid2BS.shape[0]/1000,iR/m_DisGrid2BS.shape[0], mm_RSRP))
            for jC in range(m_DisGrid2BS.shape[1]):
                d_2D=m_DisGrid2BS[iR, jC]
                s_attu=m_AntAttenuate[iR, jC]
                delta_h=self.__h_BS-self.__h_UT
                d_3D=np.sqrt(np.power(d_2D, 2)+np.power(delta_h, 2))

                nlos_Pathloss=32.4+30*math.log(d_3D, 10)+20*math.log(f_c/1e9, 10)+0

                ret_RSRP=self.__P_max+17.5-1-13-nlos_Pathloss-s_attu #+np.random.rand()*5 #15
                m_RSRP[iR,jC]=ret_RSRP

        n_m_RSRP = m_RSRP.shape[0]
        t_idx = int(n_m_RSRP/self.__n_DF)
        strIndex=['Grid_'+str(i+1) for i in range(n_m_RSRP)]
        strColum=['Cell_'+str(i+1) for i in range(m_RSRP.shape[1])]
        for iD in range(self.__n_DF):
            if iD<(self.__n_DF-1):
                mt_RSRP = np.round(m_RSRP[iD*t_idx:(iD+1)*t_idx,:], 3)
                t_strIndex = strIndex[iD*t_idx:(iD+1)*t_idx]
            else:
                mt_RSRP = np.round(m_RSRP[(iD)*t_idx:,:], 3)
                t_strIndex = strIndex[(iD)*t_idx:]
            pd_DF=pd.DataFrame(data=mt_RSRP, index=t_strIndex, columns=strColum)
            pd_DF.to_csv((self.__docHeader+'3GridRSRP'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_'),index=True)
        
        return

########################################################################################################################
    def CalLmtSINR(self):
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start)))
        for iD in range(self.__n_DF):
            strTReadCsv = (self.__docHeader+'3GridRSRP'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_')
            try:
                pd_DF = pd.concat([pd_DF, pd.read_csv(strTReadCsv)])
            except:
                pd_DF = pd.read_csv(strTReadCsv)
        m_GridRSRP0=pd_DF.values[:,1:]
        v_GridID = np.array([int(l.replace('Grid_', '')) for l in pd_DF.iloc[:,0].values]).reshape(-1,1)
        
        del pd_DF
        gc.collect()
        pd_DF=pd.read_csv((self.__docHeader+'1BSInfo.csv').replace('p/CSV_','p/net/CSV_'))
        m_BSCtnCdt=pd_DF.values
        del pd_DF
        gc.collect()
        r_GridRSRP = m_GridRSRP0.shape[0]
        v_hRSRP = np.max(m_GridRSRP0, axis=1)
        v_hIdx = np.argmax(m_GridRSRP0, axis=1)
        m_GridRSRP = m_GridRSRP0
        for iRGR in range(r_GridRSRP):
            m_GridRSRP[iRGR,v_hIdx[iRGR]]=-999
        tmp = np.copy(m_BSCtnCdt[v_hIdx, 2:5:2])
        v_hIdx_tmp =np.copy(v_hIdx.reshape(-1,1))+1
        v_hRSRP_tmp= np.copy(v_hRSRP.reshape(-1,1))
        m_LimitSINR=np.hstack((v_GridID, v_hIdx_tmp,tmp,v_hRSRP_tmp,v_hRSRP_tmp+125))
        t_idx = int(len(m_GridRSRP)/self.__n_DF)
        for i_ in range(self.__n_DF):
            if i_<(self.__n_DF-1):
                mt_GridRSRP = m_GridRSRP[i_*t_idx:(i_+1)*t_idx,:]
            else:
                mt_GridRSRP = m_GridRSRP[(i_)*t_idx:,:]
            vt_GridITF_J = (np.sum(10**(mt_GridRSRP/10),axis = 1)+10**(-125/10)).flatten()
            try:
                v_GridITF_J = np.hstack((v_GridITF_J, vt_GridITF_J))
            except:
                v_GridITF_J = vt_GridITF_J
        # v_GridITF_J = (np.sum(10**(m_GridRSRP/10),axis = 1)+10**(-125/10)).flatten()
        v_GridITF= [10*math.log(v_GridITF_J[i], 10) for i in range(len(v_GridITF_J))]
        v_GridITF=np.array(v_GridITF).reshape(-1,1)
        m_LimitSINR=np.hstack((m_LimitSINR,v_hRSRP_tmp-v_GridITF,v_GridITF,np.zeros((r_GridRSRP,1)),m_GridRSRP))

        strColum=['Grid_ID','Cell_ID','Circle_ID','Azimuth','RSRP', 'SINR_NO_ITF', 'SINR_FULL_ITF', 'FULL_ITF', '']
        for i in range(m_GridRSRP.shape[1]):
            strColum.append('Cell_'+str(i+1))

        del m_BSCtnCdt, tmp, m_GridRSRP0
        gc.collect()
        
        n_m_RSRP = m_LimitSINR.shape[0]
        t_idx = int(n_m_RSRP/self.__n_DF)
        for iD in range(self.__n_DF):
            if iD<(self.__n_DF-1):
                mt_RSRP = m_LimitSINR[iD*t_idx:(iD+1)*t_idx,:]
            else:
                mt_RSRP = m_LimitSINR[(iD)*t_idx:,:]
            # t_strIndex = strIndex[iD*t_idx:(iD+1)*t_idx]
            pd_DF=pd.DataFrame(data=mt_RSRP, columns=strColum)
            pd_DF.to_csv((self.__docHeader+'4LmtSINR'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_'),index=False)
        
        del pd_DF, m_LimitSINR
        gc.collect()

        return


########################################################################################################################
    def GetStaticQLDataSlice(self, pd0_atvCell, pd_atvCell, v_action, i_step):
        v_RSRP = pd_atvCell['RSRP'].values
        v_ITF_dec = pd0_atvCell['ITF'].values - pd_atvCell['ITF'].values
        v_SINR_dec = pd0_atvCell['SINR'].values - pd_atvCell['SINR'].values
        v_THR_dec = pd0_atvCell['THR'].values - pd_atvCell['THR'].values
        v_Satify = pd_atvCell['Satify'].values
        v_Reward = pd_atvCell['Reward'].values
        v_RSRP_oft = pd_atvCell['RSRP_oft'].values

        vt_action = [0]*self.__n_BS
        vt_RSRP_ = [0]*self.__n_BS
        vt_ITF_dec = [0]*self.__n_BS
        vt_SINR_dec = [0]*self.__n_BS
        vt_THR_dec = [0]*self.__n_BS
        vt_RSRP_oft = [0]*self.__n_BS
        vt_Satify = [0]*self.__n_BS
        vt_Reward = [0]*self.__n_BS
        v_atvCell = pd_atvCell['Cell_ID'].values - 1
        for i_ in range(len(pd_atvCell)):
            vt_action[v_atvCell[i_]] = v_action[i_]
            vt_RSRP_[v_atvCell[i_]] = v_RSRP[i_]
            vt_ITF_dec[v_atvCell[i_]] = v_ITF_dec[i_]
            vt_SINR_dec[v_atvCell[i_]] = v_SINR_dec[i_]
            vt_THR_dec[v_atvCell[i_]] = v_THR_dec[i_]
            vt_RSRP_oft[v_atvCell[i_]] = v_RSRP_oft[i_]
            vt_Satify[v_atvCell[i_]] = v_Satify[i_]
            vt_Reward[v_atvCell[i_]] = v_Reward[i_]
        st_ = 'Step'+str(np.round(0.001*i_step+self.__i_itr, 3))
        pdt_action = pd.DataFrame(data=[vt_action], index=[st_], columns=self.__all_CellID)
        pdt_RSRP = pd.DataFrame(data=[vt_RSRP_], index=[st_], columns=self.__all_CellID)
        pdt_ITF_dec = pd.DataFrame(data=[vt_ITF_dec], index=[st_], columns=self.__all_CellID)
        pdt_SINR_dec = pd.DataFrame(data=[vt_SINR_dec], index=[st_], columns=self.__all_CellID)
        pdt_THR_dec = pd.DataFrame(data=[vt_THR_dec], index=[st_], columns=self.__all_CellID)
        pdt_RSRP_oft = pd.DataFrame(data=[vt_RSRP_oft], index=[st_], columns=self.__all_CellID)
        pdt_Satify = pd.DataFrame(data=[vt_Satify], index=[st_], columns=self.__all_CellID)
        pdt_Reward = pd.DataFrame(data=[vt_Reward], index=[st_], columns=self.__all_CellID)

        if self.__pd_static_actions is None:
            self.__pd_static_actions = pdt_action 
            self.__pd_static_RSRP = pdt_RSRP
            self.__pd_static_ITF_dec = pdt_ITF_dec
            self.__pd_static_SINR_dec = pdt_SINR_dec
            self.__pd_static_THR_dec = pdt_THR_dec
            self.__pd_static_RSRP_oft = pdt_RSRP_oft
            self.__pd_static_Satify_dec = pdt_Satify
            self.__pd_static_Reward = pdt_Reward
        else:
            self.__pd_static_actions = pd.concat([self.__pd_static_actions, pdt_action], axis=0, join='inner')
            self.__pd_static_RSRP = pd.concat([self.__pd_static_RSRP, pdt_RSRP], axis=0, join='inner')
            self.__pd_static_ITF_dec = pd.concat([self.__pd_static_ITF_dec, pdt_ITF_dec], axis=0, join='inner')
            self.__pd_static_SINR_dec = pd.concat([self.__pd_static_SINR_dec, pdt_SINR_dec], axis=0, join='inner')
            self.__pd_static_THR_dec = pd.concat([self.__pd_static_THR_dec, pdt_THR_dec], axis=0, join='inner')
            self.__pd_static_RSRP_oft = pd.concat([self.__pd_static_RSRP_oft, pdt_RSRP_oft], axis=0, join='inner')
            self.__pd_static_Satify_dec = pd.concat([self.__pd_static_Satify_dec, pdt_Satify], axis=0, join='inner')
            self.__pd_static_Reward = pd.concat([self.__pd_static_Reward, pdt_Reward], axis=0, join='inner')
        
        return v_Reward, st_

########################################################################################################################
    def CalCommMetric(self, pd0_atvCell, pd_atv_GridRSRP, v_action):
        pd_atvCell = pd0_atvCell.copy()
        v_satify_THR = pd_atvCell['SatifyTHR'].values
        pd_atvCell['SINR_lb'] = np.round([10*math.log((2**(v_satify_THR[i]/10)-1),10) for i in range(len(v_satify_THR))], 2)
        pd_atvCell['RSRP'] = pd_atvCell['RSRP'].values - v_action
        v_atvCell = pd_atvCell['Cell_ID'].values
        pd_atv_GridRSRP_ = pd_atv_GridRSRP.copy()
        for i_ in range(len(pd_atvCell)):
            t_ = 'Cell_'+str(v_atvCell[i_])
            pd_atv_GridRSRP_[t_]  = pd_atv_GridRSRP[t_].values - ([v_action[i_]]*len(v_action))
        
        v_GridITF_J = np.sum(10**(pd_atv_GridRSRP.values/10),axis =1)+10**(-125/10)
        pd_atvCell['ITF'] = np.round([np.round(10*math.log(v_GridITF_J[i], 10), 3) for i in range(len(v_GridITF_J))], 2)
        pd_atvCell['SINR'] = np.round(pd_atvCell['RSRP'].values - pd_atvCell['ITF'].values, 2)
        pd_atvCell['THR'] = np.round(10*np.log2(1+10**(pd_atvCell['SINR'].values/10)), 2)        
        v_THR_oft = pd_atvCell['THR'].values - v_satify_THR
        pd_atvCell['THR_oft'] = v_THR_oft*(v_THR_oft>0).astype(np.int8)
        pd_atvCell['RSRP_lb'] = pd_atvCell['SINR_lb'].values + pd_atvCell['ITF'].values
        pd_atvCell['RSRP_oft'] = pd_atvCell['RSRP'].values - pd_atvCell['RSRP_lb'].values
        pd_atvCell['Satify'] = (np.round(pd_atvCell['RSRP_oft'].values, 0)>=0).astype(np.int8)
        try:
            pd_atvCell['Reward']
        except:
            pd_atvCell['Reward'] = [0]*len(v_action)
        
        del pd_atv_GridRSRP, v_satify_THR, v_atvCell, v_GridITF_J, v_THR_oft
        gc.collect()
        
        return pd_atvCell, pd_atv_GridRSRP_


########################################################################################################################
    def activeCellGrid(self, rt):
        if self.__pd_4LmtSINR is None:
            for iD in range(self.__n_DF):
                strTReadCsv = (self.__docHeader+'4LmtSINR'+str(iD)+'.csv').replace('p/CSV_','p/net/CSV_')
                try:
                    self.__pd_4LmtSINR = pd.concat([self.__pd_4LmtSINR, pd.read_csv(strTReadCsv)])
                except:
                    self.__pd_4LmtSINR = pd.read_csv(strTReadCsv)

        v_cellID=self.__pd_4LmtSINR.loc[:,'Cell_ID'].values
        v_gridID=self.__pd_4LmtSINR.loc[:,'Grid_ID'].values
        v_cellID_sgl=np.sort(list(set(v_cellID)))
        self.__all_CellID = ['Cell_'+str(i+1) for i in range(np.max(v_cellID_sgl))]
        self.__n_BS = int(str(self.__all_CellID[-1]).replace('Cell_',''))

        v_thrScenario=[]
        for i in range(len(v_cellID_sgl)):
            vt_gridCell=v_gridID[np.argwhere((i+1)==v_cellID).flatten()]
            if 0==len(vt_gridCell.tolist()):
                continue
            v_thrScenario.append(random.sample(self.__v_THRScenario, 1))
            t_idx = np.argwhere(random.sample(vt_gridCell.tolist(), 1)==v_gridID).flatten()
            try:
                v_atvGridIdx = np.hstack((v_atvGridIdx, t_idx))
            except:
                v_atvGridIdx = t_idx

        v_thrScenario = np.array(v_thrScenario).flatten()
        pd_allCell = self.__pd_4LmtSINR.loc[:, ['Cell_ID', 'Grid_ID', 'Circle_ID', 'Azimuth', 'RSRP']+self.__all_CellID]
        pd_allCell = pd_allCell.iloc[v_atvGridIdx,:]
        pd_allCell['SatifyTHR'] = v_thrScenario

        v_atvCell = np.sort(random.sample(v_cellID_sgl.tolist(), int(len(v_thrScenario)*rt)))
        pd_atvCell = (pd_allCell.iloc[v_atvCell-1]).copy()

        for iC_ in v_cellID_sgl:
            if iC_ not in v_atvCell:
                pd_atvCell['Cell_'+str(iC_)] = [-1001]*len(v_atvCell)
        
        pd_atv_GridRSRP = pd_atvCell[self.__all_CellID]
        pd_atvCell, pd_atv_GridRSRP = self.CalCommMetric(pd_atvCell, pd_atv_GridRSRP, [0]*len(pd_atvCell))
        
        pd_atvCell['Reward'] = [0]*len(pd_atvCell)
        
        self.__pd0_atvCell = pd_atvCell.copy()
        self.__pd0_atv_GridRSRP = pd_atv_GridRSRP.copy()
        self.__l_pd_static_table = [None]*self.__n_BS
        self.__l_pd_rawQ_table = [None]*self.__n_BS
        
        return pd_atvCell, pd_atv_GridRSRP

########################################################################################################################
    def perfGridCellID(self, pd_atvCell):        
        v_cluster_RSRP_dec = self.__v_cluster_RSRP_dec
        v_RSRP_oft = pd_atvCell['RSRP_oft'].values        
        v_atvCell = pd_atvCell['Cell_ID'].values
        # v_cell_cluster_RSRP = [0]*len(v_RSRP_oft)
        l_atv_state = [0]*len(v_RSRP_oft)
        vt_idx = np.argwhere(v_RSRP_oft<v_cluster_RSRP_dec[0]).flatten()
        l_cur_state = []
        if len(vt_idx)>0:
            st_ = 'S0|'+str(v_cluster_RSRP_dec[0]-v_cluster_RSRP_dec[1])+': '+str(v_atvCell[vt_idx]).replace('[ ','[')
            if (self.__pd_Q_table is None):
                self.__pd_Q_table = pd.DataFrame([[0]*len(self.__v_actions)], index=[st_], columns=self.__v_actions)
            else:
                if st_ not in self.__pd_Q_table.index:
                    pd_ = pd.DataFrame([[0]*len(self.__v_actions)], index=[st_], columns=self.__v_actions)
                    self.__pd_Q_table = pd.concat([self.__pd_Q_table, pd_], axis=0, join='inner')
            l_cur_state.append(st_)
            for i_ in vt_idx:
                l_atv_state[i_] = st_        
        
        for i_ in range(len(v_cluster_RSRP_dec)):            
            if (i_+1)<len(v_cluster_RSRP_dec):
                vt_idx_l = np.argwhere(v_RSRP_oft>=v_cluster_RSRP_dec[i_]).flatten()
                vt_idx_h = np.argwhere(v_RSRP_oft<v_cluster_RSRP_dec[i_+1]).flatten()
                vt_idx_l_h = np.intersect1d(vt_idx_l, vt_idx_h)
            else:
                vt_idx_l_h = np.argwhere(v_RSRP_oft>=v_cluster_RSRP_dec[i_]).flatten()
            if 0==len(vt_idx_l_h):
                continue
            st_ = 'S'+str(i_+1)+'|'+str(v_cluster_RSRP_dec[i_])+': '+str(v_atvCell[vt_idx_l_h]).replace('[ ','[')            
            for j_ in vt_idx_l_h:
                l_atv_state[j_] = st_
                # v_cell_cluster_RSRP[j_] = v_cluster_RSRP_dec[i_]
            if (self.__pd_Q_table is None):
                self.__pd_Q_table = pd.DataFrame([[0]*len(self.__v_actions)], index=[st_], columns=self.__v_actions)
            else:
                if st_ not in self.__pd_Q_table.index:
                    pd_ = pd.DataFrame([[0]*len(self.__v_actions)], index=[st_], columns=self.__v_actions)
                    self.__pd_Q_table = pd.concat([self.__pd_Q_table, pd_], axis=0, join='inner')
            l_cur_state.append(st_)
            
        pd_atvCell['State'] = l_atv_state
        pd_atvCell['CluRSRP'] = [(int(i/self.__delt)*self.__delt)*(i>0) for i in v_RSRP_oft]  #v_cell_cluster_RSRP
        self.__l_index_Q_table = self.__pd_Q_table.index.tolist()
        
        return l_cur_state, pd_atvCell

########################################################################################################################
    def QStaticLearningLoop(self):
        pd_atvCell = self.__pd0_atvCell
        pd_atv_GridRSRP = self.__pd0_atv_GridRSRP
        v_RSRP_oft = pd_atvCell['RSRP_oft'].values
        v_atvCell = pd_atvCell['Cell_ID'].values
        v_RSRP = pd_atvCell['RSRP'].values
        v_RSRP_lb = pd_atvCell['RSRP_lb'].values
        
        v_action = [0]*len(v_atvCell)
        pd_atvCell, pd_atv_GridRSRP = self.CalCommMetric(pd_atvCell, pd_atv_GridRSRP, [0]*len(v_atvCell))
        v_Reward, lt_index = self.GetStaticQLDataSlice(pd_atvCell, pd_atvCell, v_action, 0)
        
        l_index = [lt_index]

        for i_ in range(len(v_atvCell)):
            try:
                len(self.__l_pd_static_table[v_atvCell[i_]-1])
                continue
            except:
                vt_ = [v_RSRP_lb[i_], v_RSRP[i_]]
                vt_row = np.linspace(int(np.floor(np.min(vt_)))-100, int(np.ceil(np.max(vt_)))+100, int(np.ceil(np.max(vt_))-np.floor(np.min(vt_))+201))
                vt_col = np.linspace(0, 5, 6)
                self.__l_pd_static_table[v_atvCell[i_]-1] = pd.DataFrame(np.zeros((len(vt_row), len(vt_col))), index=vt_row, columns=vt_col)

        ret = self.__n_step
        v_reward_0 = [0]
        v_sum_action = [0]*len(v_atvCell)
        for i_step in range(self.__n_step):
            v_action = [0]*len(v_atvCell)
            v_RSRP = pd_atvCell['RSRP'].values
            v_RSRP_oft = pd_atvCell['RSRP_oft'].values
            for i_ in range(len(v_atvCell)):
                pd_ = self.__l_pd_static_table[v_atvCell[i_]-1]
                if v_RSRP_oft[i_]<0:
                    vt_action = [0]
                else:
                    t_ = np.min([5, np.floor(v_RSRP_oft[i_])])
                    t_ = int(np.max([0, np.min([t_, 15-v_sum_action[i_]])]))
                    vt_action = np.linspace(0, t_, t_+1).tolist()
                l_ = int(v_RSRP[i_] + 0.5)+0.0
                if (random.uniform(0, 1)>0.9) or ((pd_.loc[l_]==0).all):
                    v_action[i_] = random.sample(vt_action, 1)
                else:
                    vt_ = pd_.loc[l_, vt_action].values
                    v_action[i_] = int(vt_.idxmax())
            v_action = np.array(v_action).flatten()
            v_sum_action = v_sum_action + v_action
            pd_atvCell_, pd_atv_GridRSRP = self.CalCommMetric(pd_atvCell, pd_atv_GridRSRP, v_action)            
            v_Reward = self.__v_reward_weight[0]*v_action - self.__v_reward_weight[1]*(pd_atvCell['THR'].values - pd_atvCell_['THR'].values)
            # v_Reward = self.__v_reward_weight[0] * v_action + self.__v_reward_weight[1] *  pd_atvCell_['THR'].values
            v_reward_0 = np.hstack((v_reward_0, np.round(np.sum(v_Reward), 1)))
            pd_atvCell_['Reward'] = v_Reward
            v_Reward, lt_index = self.GetStaticQLDataSlice(pd_atvCell, pd_atvCell_, v_action, i_step+1)
            l_index.append(lt_index)
            for i_ in range(len(v_atvCell)):
                pd_ = self.__l_pd_static_table[v_atvCell[i_]-1]
                l_ = np.floor(v_RSRP[i_]) + 0.0
                Q_s_a = pd_.loc[l_, v_action[i_]]
                v_RSRP_ = pd_atvCell_['RSRP'].values
                l_ = int(v_RSRP_[i_]+0.5) + 0.0
                Q_s_1_max_a = np.max(pd_.loc[l_, :].values)
                pd_.loc[l_, v_action[i_]] += np.round(0.1*(v_Reward[i_] + 0.9*Q_s_1_max_a - Q_s_a), 3)
                self.__l_pd_static_table[v_atvCell[i_]-1] = pd_
            
            pd_atvCell = pd_atvCell_
            
            if ((self.__i_step_dur)<i_step) and (np.sum(v_reward_0[self.__i_step_dur:]==0)==self.__i_step_dur):
                ret = i_step + 1 - self.__i_step_dur
                break
        
        return ret, l_index, sum(pd_atvCell['Satify'].values), pd_atvCell

########################################################################################################################
    def QLearningLoop(self, rt):        
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start)))
        v_static_stop_step = [0]
        m_static_stat = None
        t_str = 'wt'+str(self.__v_reward_weight[0]) + '-' + str(self.__v_reward_weight[1]) + '_rt' +str(rt)
        tt_str = 'p/log/sta/'+t_str.replace('-','_')+'/CSV_'
        for itr in range(self.__n_itr): 
            self.__i_step = 0
            self.__i_itr = itr
            self.__l_step_itr = []            

            pd_atvCell, pd_atv_GridRSRP = self.activeCellGrid(rt)
            v_atvCell = pd_atvCell['Cell_ID'].values
            l_cur_state, pd_atvCell = self.perfGridCellID(pd_atvCell)
            n_Satify = np.sum(self.__pd0_atvCell['Satify'].values)
                        
            i_step_static, l_static_step_itr, n_static_satify, pd_atvCell = self.QStaticLearningLoop()
            v_static_stop_step.append(i_step_static)
            vt_sum_action = np.sum(self.__pd_static_actions.loc[l_static_step_itr].values, axis=0)
            if np.max(vt_sum_action)>15:
                self.__log_txt += (self.__pd_static_actions.to_string(index=True) +'\n')
                self.__log_txt += (self.__pd_static_actions.loc[l_static_step_itr].to_string(index=True) +'\n')
                self.__log_txt += (str(vt_sum_action) +'\n\n')
                
            pd_ = pd.DataFrame([vt_sum_action], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_actions = pd.concat([self.__pd_static_actions, pd_], axis=0, join='inner')
            vt_RSRP = np.round(self.__pd_static_RSRP.iloc[-1,:].values, 3)
            pd_ = pd.DataFrame([vt_RSRP], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_RSRP = pd.concat([self.__pd_static_RSRP, pd_], axis=0, join='inner')
            vt_ITF = np.round(np.sum(self.__pd_static_ITF_dec.loc[l_static_step_itr].values, axis=0), 3)
            pd_ = pd.DataFrame([vt_ITF], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_ITF_dec = pd.concat([self.__pd_static_ITF_dec, pd_], axis=0, join='inner')
            vt_SINR = np.round(np.sum(self.__pd_static_SINR_dec.loc[l_static_step_itr].values, axis=0), 3)
            pd_ = pd.DataFrame([vt_SINR], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_SINR_dec = pd.concat([self.__pd_static_SINR_dec, pd_], axis=0, join='inner')
            vt_THR = np.round(np.sum(self.__pd_static_THR_dec.loc[l_static_step_itr].values, axis=0), 3)
            pd_ = pd.DataFrame([vt_THR], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_THR_dec = pd.concat([self.__pd_static_THR_dec, pd_], axis=0, join='inner')
            vt_RSRP_oft = np.round(np.sum(self.__pd_static_RSRP_oft.loc[l_static_step_itr].values, axis=0), 3)
            pd_ = pd.DataFrame([vt_RSRP_oft], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_RSRP_oft = pd.concat([self.__pd_static_RSRP_oft, pd_], axis=0, join='inner')
            vt_satify = self.__pd_static_Satify_dec.iloc[-1,:].values
            vt_satify_dec = self.__pd0_atvCell['Satify'].values - pd_atvCell['Satify'].values
            t_no_satify = len(np.argwhere(vt_satify_dec>0).flatten())
            t_satify = len(np.argwhere(vt_satify_dec<0).flatten())
            pdt_ = pd.DataFrame([vt_satify], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_Satify_dec = pd.concat([self.__pd_static_Satify_dec, pdt_], axis=0, join='inner')
            vt_Reward = np.round(np.sum(self.__pd_static_Reward.loc[l_static_step_itr].values, axis=0), 3)
            pd_ = pd.DataFrame([vt_Reward], index=['Itr'+str(self.__i_itr)], columns=self.__all_CellID)
            self.__pd_static_Reward = pd.concat([self.__pd_static_Reward, pd_], axis=0, join='inner')
            
            t_avg_RSRP_0 = 10*math.log(np.sum(10**(self.__pd0_atvCell['RSRP'].values/10))/len(self.__pd0_atvCell), 10)
            t_avg_satisfy_0 = np.sum(self.__pd0_atvCell['Satify'].values)/len(self.__pd0_atvCell)
            t_avg_RSRP_oft_0 = 10*math.log(np.sum(10**(self.__pd0_atvCell['RSRP_oft'].values/10))/len(self.__pd0_atvCell), 10)
            t_avg_action = 10*math.log(np.sum(10**(vt_sum_action/10))/len(self.__pd0_atvCell), 10)
            t_avg_RSRP = 10*math.log(np.sum(10**(vt_RSRP/10))/len(self.__pd0_atvCell), 10)
            t_avg_ITF = 10*math.log(np.sum(10**(vt_ITF/10))/len(self.__pd0_atvCell), 10)
            t_avg_SINR = 10*math.log(np.sum(10**(vt_SINR/10))/len(self.__pd0_atvCell), 10)
            t_avg_THR = np.sum(vt_THR)/len(self.__pd0_atvCell)
            t_avg_Reward = np.sum(vt_Reward)/len(self.__pd0_atvCell)
            vt_ = [t_avg_RSRP_0, t_avg_satisfy_0, t_avg_RSRP_oft_0, t_avg_action, t_avg_RSRP, t_avg_ITF, t_avg_SINR, t_avg_THR, t_avg_Reward]
            vt_ = np.hstack((self.__n_BS, len(v_atvCell), i_step_static+1, np.round(vt_, 3), [np.sum(vt_satify), t_no_satify, t_satify]))
            vt_J_noatv = 10**((np.array([self.__P_max]*(self.__n_BS-len(v_atvCell))))/10)
            vt_ = np.hstack((vt_, 10*math.log(np.sum(vt_J_noatv)/self.__n_BS, 10), 10*math.log(np.sum(vt_J_noatv*0.5)/self.__n_BS, 10)))
            vt_J_proposed = 10**((vt_sum_action)/10)
            vt_ = np.hstack((vt_, 10*math.log((np.sum(vt_J_noatv)+np.sum(vt_J_proposed))/self.__n_BS, 10)))
            if m_static_stat is None:
                m_static_stat = vt_
            else:
                m_static_stat = np.vstack((m_static_stat, vt_))
            if os.path.exists('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log/sta/'+t_str.replace('-','_')):
                pass
            else:
                os.makedirs('OutCSV_C'+str(self.__circ)+'L_St'+str(self.__step)+'p/log/sta/'+t_str.replace('-','_'))
            t_ = sys.getsizeof(self.__pd_static_Satify_dec)/1e6
            if t_>50:
                self.__pd_static_actions.to_csv((self.__docHeader+t_str+'_5pd0_static_actions_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_actions = None
                self.__pd_static_RSRP.to_csv((self.__docHeader+t_str+'_5pd1_static_RSRP_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_RSRP = None                
                self.__pd_static_ITF_dec.to_csv((self.__docHeader+t_str+'_5pd2_static_ITF_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_ITF_dec = None
                self.__pd_static_SINR_dec.to_csv((self.__docHeader+t_str+'_5pd3_static_SINR_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_SINR_dec = None                
                self.__pd_static_THR_dec.to_csv((self.__docHeader+t_str+'_5pd4_static_THR_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_THR_dec = None
                self.__pd_static_RSRP_oft.to_csv((self.__docHeader+t_str+'_5pd4_static_RSRP_oft_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_RSRP_oft = None
                self.__pd_static_Satify_dec.to_csv((self.__docHeader+t_str+'_5pd5_static_Satify_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_Satify_dec = None                
                self.__pd_static_Reward.to_csv((self.__docHeader+t_str+'_5pd6_static_Reward_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
                self.__pd_static_Reward = None
                
                
                self.__i_file += 1
            plt.rcParams['font.size'] = 12
            if itr>1 and (itr%1==0):
                plt.clf()
                plt.grid(linestyle="-.", color="k", linewidth='0.5')
                self.cdfplot(v_static_stop_step, 'Static QL', 'k-', 1)
                plt.xlim([-5, self.__n_step+5]) 
                plt.ylim([-.05, 1.05])
                plt.title('[C' + str(self.__circ) + 'L/Step' + str(self.__step) + ': ' + str(itr) + '/' + str(self.__n_itr) +']: ' + t_str)
                vt_ = [np.min(v_static_stop_step[1:]), np.mean(v_static_stop_step[1:]), np.median(v_static_stop_step[1:]), np.max(v_static_stop_step[1:])]
                plt.text(self.__n_step*0.5,  0.065, 'Stat: '+str(np.round(vt_,1)))
                plt.pause(0.0001)
                # plt.legend(loc='best')
                plt.ioff()
                vt_ = [int(time.time()-self.__t_start), sys._getframe().f_lineno, np.round(itr/self.__n_itr+.0001, 4), itr, self.__n_itr]
                print(([len(v_atvCell), n_Satify, n_static_satify], vt_))
        l_str_col = ['N_Cell', 'N_atvCell', 'Iteration', 'RSRP0', 'Satify0', 'RSRP_oft0', 'act', 'RSRP', 'ITF_dec', 'SINR_dec', 'THR_dec', 'Reward', 'Satify', 'Satify-1', 'Satify+1', 'Activation', 'Sleep', 'Proposed']
        pd_static_stat = pd.DataFrame(m_static_stat, columns=l_str_col)
        pd_static_stat.to_csv((self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv'),index=True)

        self.__pd_static_actions.to_csv((self.__docHeader+t_str+'_5pd0_static_actions_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_RSRP.to_csv((self.__docHeader+t_str+'_5pd1_static_RSRP_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_ITF_dec.to_csv((self.__docHeader+t_str+'_5pd2_static_ITF_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_SINR_dec.to_csv((self.__docHeader+t_str+'_5pd3_static_SINR_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_THR_dec.to_csv((self.__docHeader+t_str+'_5pd4_static_THR_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_RSRP_oft.to_csv((self.__docHeader+t_str+'_5pd4_static_RSRP_oft_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_Satify_dec.to_csv((self.__docHeader+t_str+'_5pd5_static_Satify_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)
        self.__pd_static_Reward.to_csv((self.__docHeader+t_str+'_5pd6_static_Reward_log_'+str(self.__i_file)+'.csv').replace('p/CSV_',tt_str),index=True)

        return

########################################################################################################################
    def StatDataTackle(self):
        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        t_str= self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv'
        
        v_UE = [0.2, 0.3, 0.4, 0.5, 0.6]
        l_col = None
        for i_ in range(len(v_UE)):
            pd_ = pd.read_csv(t_str.replace('_rt0.3', '_rt'+str(v_UE[i_])))
            vt_ = pd_.values.mean(axis=0)
            try:
                m_UE = np.vstack((m_UE, vt_[1:]))
            except:
                m_UE = vt_[1:]
                l_col = pd_.columns[1:]
        pd_UE = pd.DataFrame(m_UE, columns=l_col)
        self.write2Excel('Data_stat_average.xlsx', 'UE', pd_UE, False)
        
        m_wt = np.array([[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]])
        l_row = []
        for i_ in range(len(m_wt)):
            l_row.append('('+str(m_wt[i_, 0])+', '+str(m_wt[i_, 1])+')')
            pd_ = pd.read_csv(t_str.replace('_wt'+str(v_reward_weight_rt[0])+'-'+str(v_reward_weight_rt[1]), '_wt'+str(m_wt[i_,0])+'-'+str(m_wt[i_,1])))
            vt_ = pd_.values.mean(axis=0)
            try:
                m_WT = np.vstack((m_WT, vt_[1:]))
            except:
                m_WT = vt_[1:]
        pd_WT = pd.DataFrame(m_WT, columns=l_col)
        pd_WT['Weight'] = l_row
        pd_WT = pd.concat([pd_WT['Weight'], pd_WT.drop(columns='Weight')], axis=1)
        self.write2Excel('Data_stat_average.xlsx', 'Weight', pd_WT, False)

       
        return

########################################################################################################################
    def pltPowerReduction(self):
        pd_WT = pd.read_excel(io='Data_stat_average.xlsx', sheet_name='Weight')
        pd_UE = pd.read_excel(io='Data_stat_average.xlsx', sheet_name='UE')
        
        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        t_str= self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv'
        font_size = 35
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.03995, bottom=0.125, right=0.99, top=0.905, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(30,10), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        plt.rcParams['font.size'] = font_size
        l_xlim = [str(x) for x in pd_UE['N_atvCell'].values]
        self.pltLineData(pd_UE.loc[:,['Proposed', 'Activation', 'Sleep']].values, l_xlim,
                         'The number of activated RRHs (in 1)', '(1) Power reduction for the number of activated RRHs', fig, gs, [0, 0], 1)
        self.pltLineData(pd_UE.loc[:,['RSRP_oft0', 'act', 'ITF_dec']].values, l_xlim,
                         'The number of activated RRHs (in 1)', '(2) Power offset, power reduction, and interference \n reduction for the number of activated RRHs', fig, gs, [0, 1], 0)
        plt.savefig('3Power_reduction_UE.png')
        plt.savefig((os.getcwd() + '3Power_reduction_UE.eps').replace('python', 'latex/SDQL/figures/'))
        
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.05, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(30,10), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        plt.rcParams['font.size'] = font_size
        l_xlim = [str(x) for x in pd_WT['Weight'].values]
        self.pltLineData(pd_WT.loc[:,['Proposed', 'Activation', 'Sleep']].values, l_xlim,
                         'Value of (w0, w1)', '(1) Power reduction for (w0, w1)', fig, gs, [0, 0], 1)
        self.pltLineData(pd_WT.loc[:,['RSRP_oft0', 'act', 'ITF_dec']].values, l_xlim,
                         'Value of (w0, w1)', '(2) Power offset, power reduction, and interference \n reduction for (w0, w1)', fig, gs, [0, 1], 0)
        plt.savefig('4Power_reduction_weight.png')
        plt.savefig((os.getcwd() + '4Power_reduction_weight.eps').replace('python', 'latex/SDQL/figures/'))

        return

########################################################################################################################
    def pltLineData(self, m_vals, l_xlim, s_xlable, s_title, fig, gs, v_gs, action_or):
        font_size = 35
        ax = fig.add_subplot(gs[v_gs[0], v_gs[1]])
        ax.grid(linestyle="-.", color="c", linewidth=0.001)
        x_vals = np.linspace(1, len(m_vals), len(m_vals))
        if 0==action_or:
            plt.plot(x_vals, m_vals[:,0], color='k', linestyle='-.', linewidth=2, marker='d', markersize=10, label='Power offset')
            plt.plot(x_vals, m_vals[:,1], color='k', linestyle='-', linewidth=2, marker='s', markersize=10, label='Power reduction')
            plt.plot(x_vals, m_vals[:,2], color='k', linestyle='--',  linewidth=2, marker='o', markersize=10, label='Interference reduction')
        else:
            plt.plot(x_vals, m_vals[:,0], color='k', linestyle='-', linewidth=2, marker='s', markersize=10, label='Proposed')
            plt.plot(x_vals, m_vals[:,1], color='k', linestyle='--', linewidth=2, marker='d', markersize=10, label='Activation')
            plt.plot(x_vals, m_vals[:,2], color='k', linestyle='-.',  linewidth=2, marker='o', markersize=10, label='Sleep')
        plt.ylim([-.5,25.5])
        plt.xticks(x_vals, l_xlim)
        plt.title(s_title, fontsize=font_size)
        plt.xlabel(s_xlable, fontsize=font_size)
        plt.ylabel('Value (in dB)', fontsize=font_size)
        plt.legend(loc='lower left', fontsize=font_size)

        return
    
########################################################################################################################
    def pltSatisfyTHRLossITFDec(self):
        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        t_str= self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv'
        t_idx = t_str.find('V_C')
        st_str = t_str[t_idx:t_idx+10]
        l_style = ['k-', 'r-.', 'b-', 'y-', 'b-.']        
        
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.03995, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(30,10), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)        
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        t_max = 0
        font_size = 33.7
        plt.rcParams['font.size'] = font_size
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['Satify+1'].values, 'The number of \n activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            t_max = np.max([t_max, np.max(pd_DF_ST['Satify+1'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(1) Transform from weak UE to central UE for the \n number of activated RRHs', fontsize=font_size)
        plt.xlim([-0.5, t_max+4])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('The number of transformed UE (in 1)', fontsize=font_size)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.grid(linestyle="-.", color="c", linewidth=0.001)
        m_wt = np.array([[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]])
        plt.rcParams['font.size'] = font_size
        for i_ in range(len(m_wt)):
            pd_DF_ST = pd.read_csv(t_str.replace('_wt'+str(v_reward_weight_rt[0])+'-'+str(v_reward_weight_rt[1]), '_wt'+str(m_wt[i_,0])+'-'+str(m_wt[i_,1])))
            self.cdfplot(pd_DF_ST['Satify+1'].values, 'Weights: '+str([m_wt[i_,0], m_wt[i_,1]]), l_style[i_], 1)
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title(' (2) Transform from weak UE to central UE for (w0, w1)', fontsize=font_size)
        plt.xlim([-0.5, t_max+4])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('The number of transformed UE (in 1)', fontsize=font_size)
        
        plt.savefig('6Power_satisfy.png')
        plt.savefig((os.getcwd() + '6Power_satisfy.eps').replace('python', 'latex/SDQL/figures/'))
        print((os.getcwd() + '6Power_satisfy.eps').replace('python', 'latex/SDQL/figures/'))

        
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.03995, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(30,10), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        font_size = font_size - 1
        plt.rcParams['font.size'] = font_size
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        t_max = 0
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['RSRP_oft0'].values, 'The number of activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            t_max = np.max([t_max, np.max(pd_DF_ST['RSRP_oft0'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(1) Power offset for the number of activated RRHs', fontsize=font_size)
        plt.xlim([-0.5, t_max+2.5])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('Power offset value (in dB)', fontsize=font_size)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        # t_max = 0
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['act'].values, 'The number of activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            # t_max = np.max([t_max, np.max(pd_DF_ST['act'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(2) Power reduction for the number of activated RRHs', fontsize=font_size)
        plt.xlim([-0.5, t_max+2.5])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('Power reduction value (in dB)', fontsize=font_size)
        
        plt.savefig('7Power_offset_reduction_dec.png')
        plt.savefig((os.getcwd() + '7Power_offset_reduction_dec.eps').replace('python', 'latex/SDQL/figures/'))
        print((os.getcwd() + '7Power_offset_reduction_dec.eps').replace('python', 'latex/SDQL/figures/'))
        
        
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.03995, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(30,10), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 2, figure=fig)
        plt.rcParams['font.size'] = font_size
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        # t_max = 0
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['THR_dec'].values, 'The number of activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            # t_max = np.max([t_max, np.max(pd_DF_ST['THR_dec'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(1) Throughput loss for the number of activated RRHs', fontsize=font_size)
        plt.xlim([-0.5, t_max+2.5])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('Throughput loss value (in Mb/s)', fontsize=font_size)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        # t_max = 0
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['ITF_dec'].values, 'The number of activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            # t_max = np.max([t_max, np.max(pd_DF_ST['ITF_dec'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(2) Interference reduction for the number of activated RRHs', fontsize=font_size)
        plt.xlim([-0.5, t_max+2.5])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('Interference reduction value (in dB)', fontsize=font_size)
        
        plt.savefig('8Power_THR_ITF_dec.png')
        plt.savefig((os.getcwd() + '8Power_THR_ITF_dec.eps').replace('python', 'latex/SDQL/figures/'))
        print((os.getcwd() + '8Power_THR_ITF_dec.eps').replace('python', 'latex/SDQL/figures/'))
        
        return

########################################################################################################################
    def pltDecRSRPOFTITF(self):
        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        t_str= (self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv')
        l_style = ['k:', 'r-.', 'b--', 'y-']        
        pd_DF = pd.read_csv(t_str)
        font_size = 35
        # fig = plt.figure(figsize=(12,6.0), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.0695, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(25,12), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 1, figure=fig)
        plt.rcParams['font.size'] = font_size
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        plt.plot(pd_DF['RSRP_oft0'].values, l_style[0], label='Power offset curve', linewidth = 1.0)
        plt.plot(pd_DF['act'].values, l_style[1], label='Power reduction curve', linewidth = 1.50)
        plt.plot(pd_DF['ITF_dec'].values, l_style[2], label='Interference reduction curve', linewidth = 1.0)
        plt.plot(pd_DF['THR_dec'].values, l_style[3], label='Throughput loss curve', linewidth = 1.0)
        _x = np.linspace(0, len(pd_DF['RSRP_oft0'])+20, len(pd_DF['RSRP_oft0'])+21)
        plt.plot(_x, [np.mean(pd_DF['RSRP_oft0'].values)]*len(_x), 'k-',
                 label='Average power offset: '+str(np.round(np.mean(pd_DF['RSRP_oft0'].values), 2)), linewidth = 2.5)
        plt.plot(_x, [np.mean(pd_DF['act'].values)]*len(_x), 'r-', 
                 label='Average power reduction: '+str(np.round(np.mean(pd_DF['act'].values), 2)), linewidth = 2.5)
        plt.plot(_x, [np.mean(pd_DF['ITF_dec'].values)]*len(_x), 'b-', 
                 label='Average interference reduction: '+str(np.round(np.mean(pd_DF['ITF_dec'].values), 2)), linewidth = 2.5)
        plt.plot(_x, [np.mean(pd_DF['THR_dec'].values)]*len(_x), 'y-', 
                 label='Average throughput loss: '+str(np.round(np.mean(pd_DF['THR_dec'].values), 2)), linewidth = 2.5)
        # self.cdfplot(pd_DF['RSRP_oft0'].values, 'Power offset', l_style[0], 1)
        # self.cdfplot(pd_DF['act'].values, 'Power reduction', l_style[1], 1)
        # self.cdfplot(pd_DF['ITF_dec'].values, 'Interference reduction', l_style[2], 1)
        plt.title('Power offset, power reduction, interference reduction, and throughput loss', fontsize=font_size)
        plt.xlabel('The number of samples (in 1)', fontsize=font_size)
        plt.ylabel('value', fontsize=font_size)
        plt.xlim([_x[0]-9, _x[-1]+25])
        plt.ylim([-3, 86])
        plt.legend(loc='upper center', ncol=2, fontsize=font_size)
        plt.savefig('5RSRPOft_action_ITF_analysis.png')
        plt.savefig((os.getcwd() + '5RSRPOft_action_ITF_analysis.eps').replace('python', 'latex/SDQL/figures/'))

########################################################################################################################
    def pltReward(self):
        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        tt_str = 'p/log/sta/'+t_str.replace('-','_')+'/CSV_'
        t_str= (self.__docHeader+t_str+'_5pd6_static_Reward_log_0.csv').replace('p/CSV_',tt_str)
        l_style = ['k-', 'r-.', 'b--', 'y:', 'm.-']
        pd_ST_Reward = pd.read_csv(t_str)
        l_step_itr = pd_ST_Reward['Unnamed: 0']
        t_idx = 10
        v_idx_sample = [0]
        l_idx_sample = []
        for i_ in range(len(l_step_itr)):
            if ('Step'+str(t_idx)+'.') in l_step_itr[i_]:
                if l_step_itr[i_] in l_idx_sample:
                    break
                l_idx_sample.append(l_step_itr[i_])
                v_idx_sample = np.hstack((v_idx_sample, i_))
        v_idx_sample = v_idx_sample[1:]
        pd_ST_Reward = pd_ST_Reward.loc[v_idx_sample,:]
        pd_ST_Reward['sum'] = np.sum(pd_ST_Reward.iloc[:,1:].values, axis=1)
        
        pd_ST_actions = pd.read_csv(t_str.replace('6_static_Reward', '0_static_actions'))
        pd_ST_actions = pd_ST_actions.loc[v_idx_sample,:]
        pd_ST_actions['sum'] = np.sum(pd_ST_actions.iloc[:,1:].values, axis=1)
        
        pd_ST_THR = pd.read_csv(t_str.replace('6_static_Reward', '4_static_THR'))
        pd_ST_THR = pd_ST_THR.loc[v_idx_sample,:]
        pd_ST_THR['sum'] = np.sum(pd_ST_THR.iloc[:,1:].values, axis=1)
        
        pd_ST_RSRP_oft = pd.read_csv(t_str.replace('6_static_Reward', '4_static_RSRP_oft'))
        pd_ST_RSRP_oft = pd_ST_RSRP_oft.loc[v_idx_sample,:]
        pd_ST_RSRP_oft['sum'] = np.sum(pd_ST_RSRP_oft.iloc[:,1:].values, axis=1)
        
        fig = plt.figure(figsize=(12,6.0), dpi=100, constrained_layout=False);
        plt.subplots_adjust(left=0.11, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        gs = GridSpec(1, 1, figure=fig)
        plt.rcParams['font.size'] = 23
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)        
        v_x_ = np.linspace(0, len(pd_ST_Reward)-1, len(pd_ST_Reward))
        plt.plot(v_x_, pd_ST_RSRP_oft['sum'].values, l_style[0], linewidth=3, label='RSRP offset')
        plt.plot(v_x_, pd_ST_Reward['sum'].values, l_style[1], linewidth=3, label='Reward')
        plt.plot(v_x_, pd_ST_actions['sum'].values, l_style[2], linewidth=2, label='Action')
        plt.plot(v_x_, pd_ST_THR['sum'].values, l_style[3], linewidth=2, label='Throughtput loss')
        mt_= np.vstack((pd_ST_RSRP_oft['sum'].values, pd_ST_Reward['sum'].values, pd_ST_actions['sum'].values, pd_ST_THR['sum'].values))
        plt.ylim([np.floor(np.min(mt_)/10-1)*10, np.ceil(np.max(mt_)/10+1)*10])
        plt.title('Convergence analysis')
        plt.xlabel('The number of iterations (in 1)')
        plt.ylabel('Value (in 1)')
        plt.legend(loc='upper right')
        
        plt.savefig('3convergence_analysis.png')
        plt.savefig((os.getcwd() + '3convergence_analysis.eps').replace('python', 'latex/SDQL/figures/'))
        

########################################################################################################################
    def pltIteration(self):
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.0395, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        fig = plt.figure(figsize=(25,7), dpi=100, constrained_layout=True)
        gs = GridSpec(1, 3, figure=fig)

        v_reward_weight_rt = [0.5, 0.5, 0.3]
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        tt_str = 'p/log/sta/'+t_str.replace('-','_')+'/CSV_'
        t_str= (self.__docHeader+t_str+'_5pd6_static_Reward_log_0.csv').replace('p/CSV_',tt_str)
        l_style = ['k-', 'r-.', 'b-', 'y-', 'k-.']
        pd_ST_Reward = pd.read_csv(t_str)
        l_step_itr = pd_ST_Reward['Unnamed: 0']
        t_idx = 10
        v_idx_sample = [0]
        l_idx_sample = []
        for i_ in range(len(l_step_itr)):
            if ('Step'+str(t_idx)+'.') in l_step_itr[i_]:
                if l_step_itr[i_] in l_idx_sample:
                    break
                l_idx_sample.append(l_step_itr[i_])
                v_idx_sample = np.hstack((v_idx_sample, i_))
        v_idx_sample = v_idx_sample[1:]
        pd_ST_Reward = pd_ST_Reward.loc[v_idx_sample,:]
        pd_ST_Reward['sum'] = np.sum(pd_ST_Reward.iloc[:,1:].values, axis=1)
        
        pd_ST_actions = pd.read_csv(t_str.replace('6_static_Reward', '0_static_actions'))
        pd_ST_actions = pd_ST_actions.loc[v_idx_sample,:]
        pd_ST_actions['sum'] = np.sum(pd_ST_actions.iloc[:,1:].values, axis=1)
        
        pd_ST_THR = pd.read_csv(t_str.replace('6_static_Reward', '4_static_THR'))
        pd_ST_THR = pd_ST_THR.loc[v_idx_sample,:]
        pd_ST_THR['sum'] = np.sum(pd_ST_THR.iloc[:,1:].values, axis=1)
        
        pd_ST_RSRP_oft = pd.read_csv(t_str.replace('6_static_Reward', '4_static_RSRP_oft'))
        pd_ST_RSRP_oft = pd_ST_RSRP_oft.loc[v_idx_sample,:]
        pd_ST_RSRP_oft['sum'] = np.sum(pd_ST_RSRP_oft.iloc[:,1:].values, axis=1)
        
        l_style = ['k-', 'r-.', 'b-', 'y-', 'k-.']
        t_str = 'wt'+str(v_reward_weight_rt[0]) + '-' + str(v_reward_weight_rt[1]) + '_rt' +str(v_reward_weight_rt[2])
        t_str= self.__docHeader+t_str+'_5pd0_static_iteration_stat.csv'
        t_idx = t_str.find('V_C')
        st_str = t_str[t_idx:t_idx+10]
        # fig = plt.figure(figsize=(23,6.6), dpi=100, constrained_layout=False);
        # plt.subplots_adjust(left=0.03995, bottom=0.125, right=0.985, top=0.925, wspace=0.2, hspace=0.32)
        # gs = GridSpec(1, 2, figure=fig)
        ax = fig.add_subplot(gs[0, 0])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        v_ = [0.2,0.3,0.4,0.5,0.6]
        t_max = 0
        font_size=25
        plt.rcParams['font.size'] = font_size
        for i_ in range(len(v_)):
            pd_DF_ST = pd.read_csv(t_str.replace(st_str, 'V_C'+str(self.__circ)+'L_St'+str(self.__step)+'p').replace('_rt0.3', '_rt'+str(v_[i_])))
            self.cdfplot(pd_DF_ST['Iteration'].values, 'The number of \n activated RRHs: '+str(int(pd_DF_ST.loc[0,'N_atvCell'])), l_style[i_], 1)
            t_max = np.max([t_max, np.max(pd_DF_ST['Iteration'].values)])
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(1) Iteration for the number of activated RRHs', fontsize=font_size)
        plt.xlim([8, np.ceil((t_max/10))*10+23])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('The number of iterations (in 1)', fontsize=font_size)
        
        ax = fig.add_subplot(gs[0, 1])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        m_wt = np.array([[0.1,0.9],[0.3,0.7],[0.5,0.5],[0.7,0.3],[0.9,0.1]])
        plt.rcParams['font.size'] = font_size
        for i_ in range(len(m_wt)):
            pd_DF_ST = pd.read_csv(t_str.replace('_wt'+str(v_reward_weight_rt[0])+'-'+str(v_reward_weight_rt[1]), '_wt'+str(m_wt[i_,0])+'-'+str(m_wt[i_,1])))
            self.cdfplot(pd_DF_ST['Iteration'].values, 'Weights: '+str([m_wt[i_,0], m_wt[i_,1]]), l_style[i_], 1)
        plt.legend(loc='lower right', fontsize=font_size)
        plt.title('(2) Iteration for (w0, w1)', fontsize=font_size)
        plt.xlim([8, np.ceil((t_max/10))*10+23])
        plt.ylabel('CDF (in 1)', fontsize=font_size)
        plt.xlabel('The number of iterations (in 1)', fontsize=font_size)        
        
        ax = fig.add_subplot(gs[0, 2])
        ax.grid(linestyle="-.", color="c", linewidth=0.0001)
        plt.rcParams['font.size'] = font_size   
        v_x_ = np.linspace(0, len(pd_ST_Reward)-1, len(pd_ST_Reward))
        plt.plot(v_x_, pd_ST_RSRP_oft['sum'].values, l_style[0], linewidth=3, label='Power offset')
        plt.plot(v_x_, pd_ST_actions['sum'].values, l_style[1], linewidth=3, label='Power reduction')
        plt.plot(v_x_, pd_ST_THR['sum'].values, l_style[2], linewidth=3, label='Throughtput loss')        
        plt.plot(v_x_, pd_ST_Reward['sum'].values, l_style[3], linewidth=3, label='Reward')
        mt_= np.vstack((pd_ST_RSRP_oft['sum'].values, pd_ST_Reward['sum'].values, pd_ST_actions['sum'].values, pd_ST_THR['sum'].values))
        plt.ylim([np.floor(np.min(mt_)/10-1)*10, np.ceil(np.max(mt_)/10+1)*10])
        plt.xlim([-1, 20])
        plt.title('(3) Convergence analysis', fontsize=font_size)
        plt.xlabel('The number of iterations (in 1)', fontsize=font_size)
        plt.ylabel('Value (in 1)', fontsize=font_size)
        plt.legend(loc='upper right', fontsize=font_size)
        
        plt.savefig('9iteration_analysis.png')
        plt.savefig((os.getcwd() + '9iteration_analysis.eps').replace('python', 'latex/SDQL/figures/'))
        
        return


########################################################################################################################
    def excFun(self):
        plt.close('all')
        # self.GenAntena()
        # self.NetCtnCnt()
        # self.GridBSRSRP()
        # self.CalcRSRP()
        # self.CalLmtSINR()
        
        for ir in self.__v_rt:
            self.QLearningLoop(ir)
            self.__i_file = 0

       
        self.__log_file.write(self.__log_txt)
        self.__log_file.close()
        print('%s: [C%dL/Step%.f/L%s]_[%s/%s], %.1fs/%.1fh' % (self.__curDir[-10:], self.__circ, self.__step,
                                                 sys._getframe().f_lineno, sys._getframe(0).f_code.co_name,
                                                 sys._getframe(1).f_code.co_name, (time.time()-self.__t_start), (time.time()-self.__t_start)/3600))
        print(datetime.datetime.now())


        return

########################################################################################################################

stp = 1
n_step = 100
n_itr = 1000

# v_reward_weight = [0.5, 0.5]
# v_rt = [0.2, 0.3, 0.4, 0.5, 0.6]
# for v_ in v_rt:
#     objRANInfo = RANINfo(2, stp, n_step, n_itr, v_reward_weight, [v_])
#     objRANInfo.excFun()

# v_rt = [0.3]
# m_wt = np.array([[0.1,0.9],[0.3,0.7],[0.7,0.3],[0.9,0.1]])
# for i_ in range(m_wt.shape[0]):
#     v_reward_weight = m_wt[i_,:]
#     objRANInfo3 = RANINfo(2, stp, n_step, n_itr, v_reward_weight, v_rt)
#     objRANInfo3.excFun()

v_reward_weight = [0.5, 0.5]
v_rt = [0.3]
objRANInfo = RANINfo(2, stp, n_step, n_itr, v_reward_weight, v_rt)
plt.close('all')
objRANInfo.StatDataTackle()
objRANInfo.pltPowerReduction()
objRANInfo.pltSatisfyTHRLossITFDec()
objRANInfo.pltReward()
objRANInfo.pltIteration()
objRANInfo.pltDecRSRPOFTITF()
plt.close('all')
