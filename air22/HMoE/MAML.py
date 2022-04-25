"""
元学习MAML 核心代码

loss = F.cross_entropy(y_hat, y_spt)

# loss: 求导的因变量（需要求导的函数）
# self.net.parameters(): 求导的自变量
grad = torch.autograd.grad(loss, self.net.parameters())
tuples = zip(grad, self.net.parameters()) ## 将梯度和参数\theta一一对应起来
# fast_weights这一步相当于求了一个\theta - \alpha*\nabla(L)
fast_weights = list(map(lambda p: p[1] - self.base_lr * p[0], tuples))
把网络参数作为自变量进行求导，得到每个参数的梯度，对每个参数进行更新

losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
corrects = [0 for _ in range(self.update_step + 1)]

for i in range(task_num):
    # 1. run the i-th task and compute loss for k=0
    logits = self.net(x_spt[i], vars=None, bn_training=True)
    loss = F.cross_entropy(logits, y_spt[i])
    grad = torch.autograd.grad(loss, self.net.parameters())
    fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.net.parameters())))

    # this is the loss and accuracy before first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry[i])
        losses_q[0] += loss_q

        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry[i]).sum().item()
        corrects[0] = corrects[0] + correct

    # this is the loss and accuracy after the first update
    with torch.no_grad():
        # [setsz, nway]
        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)
        loss_q = F.cross_entropy(logits_q, y_qry[i])
        losses_q[1] += loss_q
        # [setsz]
        pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
        correct = torch.eq(pred_q, y_qry[i]).sum().item()
        corrects[1] = corrects[1] + correct

    for k in range(1, self.update_step):
        # 1. run the i-th task and compute loss for k=1~K-1
        logits = self.net(x_spt[i], fast_weights, bn_training=True)
        loss = F.cross_entropy(logits, y_spt[i])
        # 2. compute grad on theta_pi
        grad = torch.autograd.grad(loss, fast_weights)
        # 3. theta_pi = theta_pi - train_lr * grad 得到下一步的权重
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

        logits_q = self.net(x_qry[i], fast_weights, bn_training=True)

        # 累加当前任务的losses_q
        loss_q = F.cross_entropy(logits_q, y_qry[i])
        losses_q[k + 1] += loss_q

        with torch.no_grad():
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
            corrects[k + 1] = corrects[k + 1] + correct





# 求导k步后所有任务的平均loss  backward
loss_q = losses_q[-1] / task_num
self.meta_optim.zero_grad()
loss_q.backward()
self.meta_optim.step()
"""
