import torch
import os
import matplotlib.pyplot as plt
os.makedirs('plots', exist_ok=True)


def distill_loss(teacher_model_outputs, student_model_outputs, labels, alpha , temperature):
    
    stud_probs = torch.nn.functional.log_softmax(student_model_outputs/temperature , dim=1)
    teacher_probs = torch.nn.functional.softmax(teacher_model_outputs/temperature , dim=1)
    kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')(stud_probs, teacher_probs) * (alpha * temperature * temperature)
    ce_loss = torch.nn.CrossEntropyLoss()(student_model_outputs, labels)* (1-alpha)
    loss = kl_div_loss + ce_loss
    return loss
               
    
def plot(x,y,ylabel,filename):
    plt.figure()
    plt.plot(x,y)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} Vs Epochs")
    plt.savefig(os.path.join('plots', filename ))
    plt.close()
    
def train(args, model, train_loader, val_loader, teacher_model=None):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    train_losses = []
    val_losses = []
    accuracies = []
    for epoch in range(args.epochs):
        train_loss, elements = 0,0
        model.train()
        for ip, mask, label in train_loader:
            ip, mask, label = ip.to(args.device), mask.to(args.device), label.to(args.device)
            optimizer.zero_grad()
            if args.mode == 'distil' or args.mode == 'rnn':
                op = model(ip)
                if teacher_model is not None:
                  
                    teacher_model_outputs = teacher_model(ip, mask)
                    loss = distill_loss(student_model_outputs=op, teacher_model_outputs=teacher_model_outputs,labels=label, alpha=0.1, temperature=5)
                else:
                    loss = criterion(op, label)
            else:
                op = model(ip,mask)
                loss = criterion(op, label)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            elements += ip.size(0)
        train_losses.append(train_loss/elements)
        
        val_loss, elements  = 0,0
        correct_preds = 0
        model.eval()
        with torch.no_grad():
            for ip, mask, label in val_loader:
                ip, mask, label = ip.to(args.device), mask.to(args.device), label.to(args.device)
                
                if args.mode == 'rnn' or args.mode == 'distil' :
                    op = model(ip)
                    if teacher_model is not None:
                      
                        teacher_model_outputs = teacher_model(ip, mask)
                        loss = distill_loss(student_model_outputs=op, teacher_model_outputs=teacher_model_outputs, alpha=0.1, labels=label,temperature=5)
                    else:
                        loss = criterion(op, label)
                else:
                    op = model(ip,mask)
                    loss = criterion(op, label)
                correct_preds += (torch.argmax(op,dim=1)==label).sum().item()
                val_loss+=loss.item()
                elements+=ip.size(0)
        val_losses.append(val_loss/elements)
        accuracy = correct_preds/elements
        accuracies.append(accuracy)
        print(f"At epoch {epoch}, Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}, Accuracy: {accuracies[-1]}")
    

    plot(range(len(train_losses)), train_losses, 'Training Loss', f'{args.mode}_Train_loss.png')
    plot(range(len(val_losses)), val_losses, 'Validation Loss', f'{args.mode}_val_loss.png')
    plot(range(len(accuracies)), accuracies, 'Accuracy', f'{args.mode}accuracy.png')
    


