import matplotlib.pyplot as plt

def make_ROC_HEP(roc_auc,top_eff,qcd_eff,thresholds,filename_ROC,filename_RvsEff):
    fig = plt.figure(figsize=(5, 5))
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(qcd_eff,top_eff,label = "Result (AUC = {0:.2f})".format(roc_auc) )
    plt.plot([0, 1], [0, 1], "k--", label="Baseline (AUC = 0.5)")
    plt.xlim(0.,1.)
    plt.ylim(0.,1.)
    plt.legend(loc=4)
    plt.grid()
    #plt.plot(qcd_eff,top_eff)
    #fig.show()
    #fig.savefig("ROC_tag3.png")
    fig.savefig(filename_ROC)
    #
    #
    #if qcd_eff[0] < 1./1e+6:
    #    qcd_eff[0] = 1./1e+6
    #qcd_eff = np.where(qcd_eff < 1./1e+6, 1./1e+6, qcd_eff)
    qcd_eff[qcd_eff < 1./1e+6] = 1./1e+6
    qcd_R = 1./qcd_eff
    fig2 = plt.figure(figsize=(5, 5))
    plt.title('Rejection vs eff')
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background rejection')
    plt.yscale("log")
    plt.xlim(0.,1.)
    plt.ylim(4,40000)
    plt.minorticks_on()
    #plt.grid()
    plt.grid(which="major", color="black", alpha=0.5)
    plt.grid(which="minor", color="gray", linestyle=":")
    plt.plot(top_eff,qcd_R)
    #fig.show()
    #fig2.savefig("RvsEff_tag3.png")
    fig2.savefig(filename_RvsEff)
    #
    print("TP:",len(top_eff),top_eff)
    print("Re:",len(qcd_R),qcd_R)
    #
    cutXX = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    isXX = [0] * 9
    for i in range(len(top_eff)):
        for j in range(len(isXX)):
            if not isXX[j] and top_eff[i] >= cutXX[j]:
                isXX[j] = 1
                print("@Top eff=",cutXX[j]*100,"%:",i,top_eff[i],qcd_R[i],thresholds[i])

def make_plot_loss(train_loss_list,val_loss_list,test_loss_list,filename,test_epoch=-1):
    epochs = len(train_loss_list)
    fig = plt.figure(figsize=(5, 5))
    #plt.axes().set_aspect('equal')
    plt.title('Loss: Train, Val, and Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0.,epochs+2)
    #plt.ylim(0.,1.02)
    #plt.grid()
    #fig.show()
    plt.plot(range(1,epochs+1),train_loss_list, color='blue',linestyle='-', label='Train_Loss')
    plt.plot(range(1,epochs+1),val_loss_list, color='red',linestyle='--', label='Val_Loss')
    if test_epoch == -1:
        plt.plot(epochs+1,test_loss_list[0], marker='.', label='Test_Loss')
    else:
        plt.plot(test_epoch,test_loss_list[0], marker='.', label='Test_Loss')
    plt.legend()
    fig.savefig(filename)
