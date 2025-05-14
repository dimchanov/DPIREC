def train(model: ResNet, optimizer: optim.Optimizer, loss_function: nn.Module, dataloader: DataLoader, relevancer, scheduler = None):

    epoch_loss, epoch_accuracy = 0, 0

    time_start = time.time()
    model.train()
    pbar = tqdm(dataloader, total=len(dataloader), leave=False)

    for batch_index, (images, labels) in enumerate(pbar, start = 0):
        # if batch_index == 100:
        #     break

        labels = labels.to(DEVICE)
        images = images.to(DEVICE)
            
        if relevancer and relevancer.update_mask:
            if batch_index + 50 == len(dataloader):
                relevancer.update_channels_masks()
                relevancer.delete_mask(True)
                relevancer.apply_mask()
                relevancer = None
            elif batch_index % 50 == 0:
                relevancer.update_channels_masks()
            
        if batch_index % 403 == 0:
            count_parameters(model)

        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        if relevancer:
            relevancer.step(loss.item())

        epoch_loss += loss.item()
        epoch_accuracy += float(torch.sum(outputs.argmax(dim=1) == labels))

        # relevancer._del_mask(model)
        torch.cuda.empty_cache()

    if relevancer:
        relevancer.update_channels_relevance_values()
        relevancer.delete_mask(True)
        relevancer.apply_mask()

    time_finish = time.time()

    epoch_loss /= len(dataloader)
    epoch_accuracy /= len(dataloader.dataset)
    return epoch_loss, epoch_accuracy, time_finish - time_start




for epoch in range(1, EPOCH + 1):
    if nn_relevancer and (len(prune_milestone) == 0 or epoch <= 10):
        nn_relevancer.update_mask = False
    elif nn_relevancer:
        nn_relevancer.update_mask = True

    # train
    if len(prune_milestone) == 0:
        train_epoch_loss, train_epoch_accuracy, train_elapsed_time = train(nn_model, optimizer, loss_function, cifar.train_dataloader, None)
    else:
        train_epoch_loss, train_epoch_accuracy, train_elapsed_time = train(nn_model, optimizer, loss_function, cifar.train_dataloader, nn_relevancer)
    print('Epoch: {:3d}; Loss [Train]: {:.6f}; Accuracy [Train]: {:.6f}; Elapsed time: {:.6f}; lr: {:.6f};'.format(
        epoch, train_epoch_loss, train_epoch_accuracy, train_elapsed_time, scheduler.get_last_lr()[0]
    ))
    # scheduler step
    scheduler.step()

    try:
        writer.add_scalar("Loss [Train]", train_epoch_loss, epoch)
        writer.add_scalar("Accuracy [Train]", train_epoch_accuracy, epoch)
    except:
        pass
    train_epochs.append(epoch)
    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)

    # print(nn_relevancer.q_list[0])
    # print(nn_relevancer.r_list[0])

    # valid
    if epoch % 3 == 1 or epoch == EPOCH:
        test_epoch_loss, test_epoch_accuracy, test_elapsed_time = valid(nn_model, loss_function, cifar.test_dataloader, nn_relevancer)
        print('\nEpoch: {}; Loss [Test]: {:.6f}; Accuracy [Test]: {:.6f}; Elapsed time: {:.6f}\n'.format(epoch, test_epoch_loss, test_epoch_accuracy, test_elapsed_time))
        test_epoch.append(epoch)
        test_losses.append(test_epoch_loss)
        test_accuracies.append(test_epoch_accuracy)
        try:
            writer.add_scalar("Loss [Test]", test_epoch_loss, epoch)
            writer.add_scalar("Accuracy [Test]", test_epoch_accuracy, epoch)
        except:
            pass

    # prune
    if len(prune_milestone) > 0 and epoch == prune_milestone[0]:
        print('!' * 100)
        # print(nn_relevancer.q_list[0])
        # print(nn_relevancer.r_list[0])
        # nn_relevancer._del_mask(nn_model)
        # count_parameters(nn_model)
        if nn_relevancer:
            nn_relevancer.drop_by_probability(prune_ratio=PRUNE_RATIO, kind="straight")
        
        # nn_relevancer.apply_mask(nn_model)
        count_parameters(nn_model)
        print('!' * 100)

        prune_milestone.pop(0)
        # if len(prune_milestone) == 0:
        #     nn_relevancer.collect = False

    torch.cuda.empty_cache()
    if nn_relevancer:
        nn_relevancer.apply_mask()

