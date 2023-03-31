function Fun00_GenerateAntenna()

clear;
close all;
clc;
delete *AntPtn_*.dB.csv;
GenerateAntenna=0

funName=mfilename;
FIdx=find('F'==funName);
funNameIdx=funName([FIdx+3:FIdx+4]);

h_BS=25;
Radius=200;
f=3e10;
lamda=(3e8)/f;
beta=2.*pi/lamda;
maxGain=40;
n=8;    m=8;
v_RowCol=[360 181];
t=linspace(0.0001,2*pi-.0001,v_RowCol(1));
d=lamda/4; 
W=beta.*d.*cos(t);
z11=((n/2).*W)-n/2*beta*d;
z12=((1/2).*W)-1/2*beta*d;
F1=sin(z11)./(n.*sin(z12));
K1=abs(F1)*maxGain/max(abs(F1));

t=linspace(0.0001,2*pi-.0001,v_RowCol(2));
W=beta.*d.*cos(t);
d=lamda/4;
z21=((m/2).*W)-m/2*beta*d;
z22=((1/2).*W)-1/2*beta*d;
F2=sin(z21)./(m.*sin(z22));
K2=abs(F2);

K0=kron(K1',K2);

v_VAntenna=K0(1,:);
[minVal minIdx]=max(K0(1,:));
[minVal3dB minIdx3dB]=min(abs(K0(1,:)-37));
dgr_Antenna3dB=abs(minIdx-minIdx3dB);
dgr_Down_Antenna=dgr_Antenna3dB+round(180*atan(h_BS/Radius)/pi);
K0=circshift(K0,[0 dgr_Down_Antenna]);

csvwrite(['CSV' funNameIdx '_AntPtn_' num2str(n) 'T' num2str(m) 'R_' num2str(maxGain) 'dB.csv'], K0)

end