function [A,k,Vehicle_cap]=text_read_a(i) 
    switch i 
    case 1 
    A=textread('X-n101-k25-q206-27591.vrp.txt'); 
    k=25; 
    Vehicle_cap=206; 
    case 2 
    A=textread('X-n115-k10-q169-12747.vrp.txt'); 
    k=10; 
    Vehicle_cap=169; 
    case 3 
    A=textread('X-n129-k18-q39-28940.vrp.txt'); 
    k=18; 
    Vehicle_cap=39; 
    case 4 
    A=textread('X-n134-k13-643-10916.vrp.txt'); 
    k=13; 
    Vehicle_cap=643; 
    case 5 
    A=textread('X-n143-k7-q1190-15700.vrp.txt'); 
    k=7; 
    Vehicle_cap=1190; 
    case 6 
    A=textread('X-n157-k13-q12-16876.vrp.txt'); 
    k=13; 
    Vehicle_cap=12; 
    case 7 
    A=textread('X-n162-k11-q1174-14138.vrp.txt'); 
    k=11; 
    Vehicle_cap=1174; 
    case 8 
    A=textread('X-n176-k26-q142-47812.vrp.txt'); 
    k=26; 
    Vehicle_cap=142; 
    case 9 
    A=textread('X-n186-k15-q974-24145.vrp.txt'); 
    k=15; 
    Vehicle_cap=974; 6.F 
    
    case 10 
    A=textread('X-n190-k8-q138-16980.vrp.txt'); 
    k=8; 
    Vehicle_cap=138; 
    case 11 
    A=textread('X-n209-k16-q101-30656.vrp.txt'); 
    k=16; 
    Vehicle_cap=101; 
    case 12 
    A=textread('X-n214-k11-q944-10856.vrp.txt'); 
    k=11; 
    Vehicle_cap=944; 
    case 13 
    A=textread('X-n228-k23-q154-25742.vrp.txt'); 
    k=23; 
    Vehicle_cap=154; 
    case 14 
    A=textread('X-n233-k16-q631-19230.vrp.txt'); 
    k=16; 
    Vehicle_cap=631; 
    case 15 
    A=textread('X-n242-k48.-q28-82751vrp.txt'); 
    k=48; 
    Vehicle_cap=28; 
    case 16 
    A=textread('X-n261-k13-1081-26558.vrp.txt'); 
    k=13; 
    Vehicle_cap=1081; 
    case 17 
    A=textread('X-n284-k15-q109-20215.vrp.txt'); 
    k=15; 
    Vehicle_cap=109; 
    case 18 
    A=textread('X-n308-k13-q246-94044.vrp.txt'); 
    k=13; 
    Vehicle_cap=246; 
    case 19 
    A=textread('X-n313-k71-q248-94044.vrp.txt'); 
    k=71; 
    Vehicle_cap=248; 
    case 20 
    A=textread('X-n317-k53-q6-78355.vrp.txt'); 
    k=53; 
    Vehicle_cap=6; 
    case 21 
    A=textread('X-n327-k20-q128-27532.vrp.txt'); 
    k=20; 
    Vehicle_cap=128; 
    case 22 
    A=textread('X-n331-k15-q23-31102.vrp.txt'); 
    k=15; 
    Vehicle_cap=23; 
    case 23 
    A=textread('X-n376-k94-q4-147713.vrp.txt'); 
    k=94; 
    Vehicle_cap=4; 
    case 24 
    A=textread('X-n401-k29-745-66187.txt'); 
    k=29; 
    Vehicle_cap=745; 
    case 25 6.G 
    
    A=textread('X-n411-k19-216-19718.txt'); 
    k=19; 
    Vehicle_cap=216; 
    case 26 
    A=textread('X-n439-k37-12-36391.txt'); 
    k=37; 
    Vehicle_cap=12; 
    case 27 
    A=textread('X-n449-k29-777-55269.txt'); 
    k=29; 
    Vehicle_cap=777; 
    case 28 
    A=textread('X-n469-k138-256-221824.txt'); 
    k=138; 
    Vehicle_cap=256; 
    case 29 
    A=textread('X-n491-k59-428-66510.txt'); 
    k=59; 
    Vehicle_cap=428; 
    case 30 
    A=textread('X-n502-k39-q13-69230.vrp.txt'); 
    k=39; 
    Vehicle_cap=13; 
    case 31 
    A=textread('X-n513-k21-142-24201.txt'); 
    k=21; 
    Vehicle_cap=142; 
    case 32 
    A=textread('X-n536-k96-371-94988.txt'); 
    k=96; 
    Vehicle_cap=371; 
    case 33 
    A=textread('X-n548-k50-11-86700.txt'); 
    k=50; 
    Vehicle_cap=11; 
    case 34 
    A=textread('X-n561-k42-74-42722.txt'); 
    k=42; 
    Vehicle_cap=74; 
    case 35 
    A=textread('X-n573-k30-210-50717.txt'); 
    k=30; 
    Vehicle_cap=210; 
    case 36 
    A=textread('X-n613-k62-523-59556.txt'); 
    k=62; 
    Vehicle_cap=523; 
    case 37 
    A=textread('X-n627-k43-110-62210.txt'); 
    k=43; 
    Vehicle_cap=110; 
    case 38 
    A=textread('X-n641-k35-1381-63737.txt'); 
    k=35; 
    Vehicle_cap=1381; 
    case 39 
    A=textread('X-n655-k131-5-106780.txt'); 
    k=131; 
    Vehicle_cap=5; 
    case 40 
    A=textread('X-n670-k130-129-146476.txt'); 
    k=130; 
    Vehicle_cap=129; 
    case 41 
    A=textread('X-n685-k75-408-68261.txt'); 
    k=75; 
    Vehicle_cap=408; 
    case 42 
    A=textread('X-n701-k44-87-81934.txt'); 
    k=44; 
    Vehicle_cap=87; 
    case 43 
    A=textread('X-n716-k35-1007-43414.txt'); 
    k=35; 
    Vehicle_cap=1007; 
    case 44 
    A=textread('X-n733-k159-25-136250.txt'); 
    k=159; 
    Vehicle_cap=25; 
    case 45 
    A=textread('X-n749-k98-396-77365.txt'); 
    k=98; 
    Vehicle_cap=396; 
    case 46 
    A=textread('X-n766-k71-166-114525.txt'); 
    k=71; 
    Vehicle_cap=166; 
    case 47 
    A=textread('X-n783-k48-832-72445.txt'); 
    k=48; 
    Vehicle_cap=832; 
    case 48 
    A=textread('X-n801-k40-20-73331.txt'); 
    k=40; 
    Vehicle_cap=20; 
    case 49 
    A=textread('X-n837-k142-44-193809.txt'); 
    k=142; 
    Vehicle_cap=44; 
    case 50 
    A=textread('X-n856-k95-9-89002.txt'); 
    k=95; 
    Vehicle_cap=9; 
    case 51 
    A=textread('X-n876-k59-764-99331.txt'); 
    k=59; 
    Vehicle_cap=764; 
    case 52 
    A=textread('X-n895-k37-1816-53946.txt'); 
    k=37; 
    Vehicle_cap=1816; 
    case 53 
    A=textread('X-n916-k207-33-329247.txt'); 
    k=207; 
    Vehicle_cap=33; 
    case 54 
    A=textread('X-n936-k151-138-132923.txt'); 
    k=151; 
    Vehicle_cap=138; 
    case 55 6.I 
    
    A=textread('X-n957-k87-11-85478.txt'); 
    k=87; 
    Vehicle_cap=11; 
    case 56 
    A=textread('X-n979-k58-998-119008.txt'); 
    k=58; 
    Vehicle_cap=998; 
    case 57 
    A=textread('X-n1001-k43-131-72402.txt'); 
    k=43; 
    Vehicle_cap=131; 
    end 
    text_read_b.m 
    function [A,k,Vehicle_cap]=text_read_b(i) 
    switch i 
    case 1 
    A=textread('Leuven2-n4000-q150-112998.txt'); 
    k=46; 
    Vehicle_cap=150; 
    case 2 
    A=textread('Antwerp2-n7000-q100-296055.vrp.txt'); 
    k=120; 
    Vehicle_cap=100; 
    case 3 
    A=textread('Ghent2-n11000-q170-264512.vrp.txt'); 
    k=110; 
    Vehicle_cap=170; 
    case 4 
    A=textread('Brussels2-n16000-q150-355779.vrp.txt'); 
    k=182; 
    Vehicle_cap=150; 
    end 
    initialization.m 
    function Positions=initialization(SearchAgents_no,dim,ub,lb) 
    Boundary_no= size(ub,2); % numnber of boundaries 
    % If the boundaries of all variables are equal and user enter a signle 
    % number for both ub and lb 
    if Boundary_no==1 
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb; 
    end 
    % If each variable has a different lb and ub 
    if Boundary_no>1 
    for i=1:dim 
    ub_i=ub(i); 
    lb_i=lb(i); 
    Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i; 
    end 
    end 
    