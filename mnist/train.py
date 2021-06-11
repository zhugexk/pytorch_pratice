import torch


def train_step(epoch, network, device, optimizer, loss_fn, train_losses, log_interval, train_loader,
               is_supervised=True):
    network.train()
    for batch_idx, (input_x, target) in enumerate(train_loader):
        input_x = input_x.to(device)
        target = target.to(device) if is_supervised else input_x.to(device)
        optimizer.zero_grad()
        output = network(input_x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}({:0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def s_train_step(epoch, network, device, optimizer, loss_fn, train_losses, log_interval, train_loader,
                 is_supervised=True):
    network.train()
    for batch_idx, (input_x, target) in enumerate(train_loader):
        input_x = input_x.to(device)
        target = target.to(device) if is_supervised else input_x.to(device)
        for parameter in network.parameters():
            for idx in range(len(parameter.view(-1))):
                optimizer.zero_grad()
                output = network(input_x)
                loss = loss_fn(output, target)
                loss.backward()
                mask = torch.zeros_like(parameter.view(-1), dtype=torch.float32)
                mask[idx] = 1.
                parameter.grad = parameter.grad * mask.view(parameter.grad.shape)
                optimizer.step()

        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}({:0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input_x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(device, network, loss_fn, test_losses, test_loader, is_supervised=True):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for input_x, target in test_loader:
            input_x = input_x.to(device)
            target = target.to(device) if is_supervised else input_x.to(device)
            output = network(input_x)
            test_loss += loss_fn(output, target).item() * test_loader.batch_size
            if is_supervised:
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    if is_supervised:
        print('\ntest set: Avg. loss: {:.4f}, Accuracy:{}/{} ({:0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    else:
        print('\ntest set: Avg. loss: {:.4f}\n'.format(test_loss))

def train(config):
    for epoch in range(1, config.n_epoch + 1):
        train_step(epoch=epoch,
                   network=config.network,
                   device=config.device,
                   optimizer=config.optimizer,
                   loss_fn=config.loss_fn,
                   train_losses=config.train_losses,
                   log_interval=config.log_interval,
                   train_loader=config.train_loader,
                   is_supervised=config.is_supervised)
        test(device=config.device,
             network=config.network,
             loss_fn=config.loss_fn,
             test_losses=config.test_losses,
             test_loader=config.test_loader,
             is_supervised=config.is_supervised)
    config.save()


def s_train(config):
    for epoch in range(1, config.n_epoch + 1):
        s_train_step(epoch=epoch,
                     network=config.network,
                     device=config.device,
                     optimizer=config.optimizer,
                     loss_fn=config.loss_fn,
                     train_losses=config.train_losses,
                     log_interval=config.log_interval,
                     train_loader=config.train_loader,
                     is_supervised=config.is_supervised)
        test(device=config.device,
             network=config.network,
             loss_fn=config.loss_fn,
             test_losses=config.test_losses,
             test_loader=config.test_loader,
             is_supervised=config.is_supervised)
    config.save()
