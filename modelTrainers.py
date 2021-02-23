import time
import torch
import numpy as np
import gc

def torch_trainer(model, optimizer, train_dataloader, loss_fn, device, epochs, val_dataloader=None ):
    best_accuracy = 0
    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
    print("-" * 60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking loss
        total_loss = 0
        t0_epoch = time.time()
        # Put the model into the training mode
        model.train()

        for batch in train_dataloader:
            b_input_ids, b_labels = batch[0],batch[1]
            # Load batch to GPU
            b_labels = b_labels.type(torch.LongTensor)

            b_input_ids, b_labels = b_input_ids.to(device), b_labels.type(torch.LongTensor).to(device)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids)

            b_labels = b_labels.long()
            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)

            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()



        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        # =======================================
        #               Evaluation
        # =======================================
        if val_dataloader is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy, preds = evaluate(model, val_dataloader, device)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | { val_loss: ^ 10.6f} | {val_accuracy: ^ 9.2f} | {time_elapsed: ^ 9.2f}")
        else:
                        # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | { 0.0: ^ 10.6f} | {0.0: ^ 9.2f} | {time_elapsed: ^ 9.2f}")
    print("\n")
    # print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")
    # return model

def evaluate(model,  val_dataloader, loss_fn, device):
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    Predictions = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        gc.collect()
        torch.cuda.empty_cache()
        # Load batch to GPU
        b_input_ids, b_labels = batch[0].to(device), batch[1].to(device)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        Predictions.extend(list(preds))
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy, Predictions
def bert_evaluate(model,  val_dataloader, loss_fn, device):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []
    Predictions = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        gc.collect()
        torch.cuda.empty_cache()
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        Predictions.extend(list(preds))
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy, Predictions
def bert_trainer(model, train_dataloader, loss_fn,optimizer, scheduler,device, val_dataloader=None, epochs=4, evaluation=False):
# def bert_trainer(model, train_dataloader, loss_fn, optimizer,device, val_dataloader=None, epochs=4,evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(
                    f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)

        print("-" * 70)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy,_ = bert_evaluate(model, val_dataloader, device, loss_fn)

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch

            print(
                f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")

    print("Training complete!")


def bert_predict(model, test_dataloader,device):

    model.eval()

    predictions = []
    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]


        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
            preds = torch.argmax(logits, dim=1).flatten()
        predictions.extend(preds)

    return predictions