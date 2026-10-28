import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from datasets import Features, Value, load_dataset
from accelerate.utils.memory import clear_device_cache
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, get_scheduler, logging
logging.set_verbosity_error()

checkpoint = "distilbert/distilbert-base-uncased"
tokeniser = AutoTokenizer.from_pretrained(checkpoint)
dataset = load_dataset("yelp_review_full").cast(Features({"label": Value("float"), "text": Value("string")}))
train_dataset = dataset["train"].shuffle(seed=0).select(range(100))
test_dataset = dataset["test"].shuffle(seed=0).select(range(100))
train_dataset = train_dataset.map(lambda x: tokeniser(x["text"], padding="max_length", truncation=True), batched=True)
test_dataset = test_dataset.map(lambda x: tokeniser(x["text"], padding="max_length", truncation=True), batched=True)

torch_train_dataset = train_dataset.remove_columns(["text"]).rename_column("label", "labels")
torch_test_dataset = test_dataset.remove_columns(["text"]).rename_column("label", "labels")
torch_train_dataset.set_format("torch")
torch_test_dataset.set_format("torch")
train_dataloader = DataLoader(torch_train_dataset, shuffle=True, batch_size=8)
test_dataloader = DataLoader(torch_test_dataset, shuffle=True, batch_size=8)

def hugging_face_run():
    # Task 1
    # Training and evaluation with Hugging Face
    
    print("\nTASK 1: hugging_face_run\n")
    # regrssion model
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="regression",
        num_labels=1
    )

    # freezing base model unless the regressor
    for param in model.base_model.parameters():
        param.requires_grad = False
    
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True

    # training arguments
    training_args = TrainingArguments(
        num_train_epochs=20, # Nº of epoch
        eval_strategy="epoch", # to evaluate at the end of the epoch
        dataloader_pin_memory=False, # working in cpu
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_strategy="no",
        output_dir="./results",
        logging_steps=5000
    )

    # metrics function
    def metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions.squeeze()
        labels = labels.squeeze()
        mse = mean_squared_error(labels, predictions)
        return {"eval_mse": mse}

    # trainer declaration
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch_train_dataset,
        eval_dataset=torch_test_dataset,
        compute_metrics=metrics,
    )

    # training and evaluation
    trainer.train()
    eval_results = trainer.evaluate()
    print("\nTASK 1: hugging_face_run | Final evaluation MSE:", eval_results["eval_mse"])



def pytorch_run():
    # Task 2
    # Training and evaluation with PyTorch
    print("\nTASK 2: pytorch_run\n")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model declaration
    my_model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        problem_type="regression",
        num_labels=1
    ).to(device)

    # Freezing the model
    for param in my_model.base_model.parameters():
        param.requires_grad = False

    # unFreezing the classifier
    for name, param in my_model.named_parameters():
        if "classifier" in name or "pre_classifier" in name:
            param.requires_grad = True

    # loss
    loss_fn = torch.nn.MSELoss()

    
    # optimizer
    optimizer = AdamW(my_model.parameters(), lr=5e-6, weight_decay=5e-6) # AdamW optimizer
    num_epochs = 20
    steps_training=num_epochs * len(train_dataloader)

    scheduler = get_scheduler(  #scheduler definition
        name="linear",
        optimizer=optimizer,
        num_training_steps=num_epochs * len(train_dataloader),
        num_warmup_steps=int(0.12 * steps_training), #warmup
    )

    # training loop
    for epoch in range(1, num_epochs + 1):
        my_model.train()
        total_loss = 0

        for batch in train_dataloader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            outputs = my_model(**batch)
            predictions = outputs.logits.squeeze()
            labels = batch["labels"].float()

            loss = loss_fn(predictions, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_dataloader)

        # evaluation for each epoch
        my_model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_dataloader:
                
                for k, v in batch.items():
                    batch[k] = v.to(device)   
                                 
                outputs = my_model(**batch)
                preds = outputs.logits.squeeze().cpu().numpy()
                labels = batch["labels"].cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels)

        mse = mean_squared_error(all_labels, all_preds)
        print(f"Epoch {epoch} - Train Loss: {avg_train_loss:.4f} | Evaluation MSE: {mse:.4f}")

    print("\nTASK 2: pytorch_run | Final evaluation MSE:", mse)



if __name__ == "__main__":

    # Task 1 test code
    hugging_face_run()


    # Task 2 test code
    
    clear_device_cache()
    print("-------------------------------------------------------------")
    pytorch_run()
