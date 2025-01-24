
# Импорт необходимых библиотек
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torchviz import make_dot
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from tqdm.notebook import tqdm
import seaborn as sns 
import pandas as pd
import os
from torchvision.utils import make_grid
from torchvision import models

# ===================================================
#  乱数固定用関数 (学籍番号下3桁を指定など)
# ===================================================
def torch_seed(seed=37):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# ===================================================
#  学習(がくしゅう)用関数 - fit_model
#   ・モデル名を引数に追加
#   ・学習履歴を返却 (epoch, train_loss, train_acc, val_loss, val_acc)
#   ・最後に学習結果を簡易表示
# ===================================================
def fit_model(model_name, net, optimizer, criterion, 
              num_epochs, train_loader, test_loader, device, 
              seed=37):
    """
    各種(かくしゅ)モデルを学習して履歴を返す関数。
    historyの各列は以下の通り:
      history[:,0] => epoch番号(1~)
      history[:,1] => train_loss
      history[:,2] => train_acc
      history[:,3] => val_loss
      history[:,4] => val_acc
    """
    torch_seed(seed)  # 乱数固定(らんすうこてい) - 学籍番号など使う

    history = []
    # -----------------------------------------------------
    # epoch(えぽっく)のループ
    # -----------------------------------------------------
    for epoch in range(num_epochs):
        # 1エポックあたりの精度(せいど)計算用
        n_train_acc, n_val_acc = 0, 0
        # 1エポックあたりの累積(るいせき)損失(そんしつ)
        train_loss, val_loss = 0.0, 0.0
        # データ件数(けんすう)
        n_train, n_test = 0, 0

        # ======== 訓練(くんれん)フェーズ ========
        net.train()
        for inputs, labels in tqdm(train_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} (Train)"):
            # バッチサイズ
            batch_size = len(labels)
            n_train += batch_size

            # GPUヘ転送
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 勾配(こうばい)の初期化(しょきか)
            optimizer.zero_grad()

            # 順伝播(じゅんでんぱ) + 損失(そんしつ)計算
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 逆伝播(ぎゃくでんぱ) + パラメータ更新
            loss.backward()
            optimizer.step()

            # 予測ラベル算出
            predicted = torch.max(outputs, 1)[1]

            # 損失合計と正解数(せいかいすう)合計
            train_loss += loss.item() * batch_size
            n_train_acc += (predicted == labels).sum().item()

        # ======== 評価(ひょうか)フェーズ ========
        net.eval()
        with torch.no_grad():
            for inputs_test, labels_test in tqdm(test_loader, desc=f"[{model_name}] Epoch {epoch+1}/{num_epochs} (Eval)"):
                test_batch_size = len(labels_test)
                n_test += test_batch_size

                inputs_test = inputs_test.to(device)
                labels_test = labels_test.to(device)

                outputs_test = net(inputs_test)
                loss_test = criterion(outputs_test, labels_test)

                # 予測(よそく)
                predicted_test = torch.max(outputs_test, 1)[1]

                # 損失合計と正解数合計
                val_loss += loss_test.item() * test_batch_size
                n_val_acc += (predicted_test == labels_test).sum().item()

        # ======== 損失・精度の算出 ========
        avg_train_loss = train_loss / n_train
        avg_val_loss   = val_loss / n_test
        train_acc      = n_train_acc / n_train
        val_acc        = n_val_acc / n_test

        print(f"[{model_name}] Epoch {epoch+1}/{num_epochs} -> "
              f"train_loss: {avg_train_loss:.5f}, train_acc: {train_acc:.5f}, "
              f"val_loss: {avg_val_loss:.5f}, val_acc: {val_acc:.5f}")

        history.append([epoch+1, avg_train_loss, train_acc, avg_val_loss, val_acc])

    history = np.array(history)
    # ===========================================================
    # 学習終了後(しゅうりょうご)に最終(さいしゅう)損失・精度だけまとめて表示
    # ===========================================================
    print(f"\n◆◆ [{model_name}] 最終結果 ◆◆")
    print(f"   [Epoch {num_epochs}] val_loss: {history[-1,3]:.5f}, val_acc: {history[-1,4]:.5f}\n")

    return history

# ===================================================
#  単一(たんいつ)モデルの学習履歴(りれき)を可視化
#  (今回は複数(ふくすう)モデルを比較(ひかく)するので拡張版を別途用意)
# ===================================================
def evaluate_history_single(history, model_name="Model"):
    """
    受け取ったhistory( shape: [epoch, 5] )を元に、
    train_loss, val_loss, train_acc, val_acc の学習曲線をプロット。
    """
    epochs = history[:,0]
    train_loss = history[:,1]
    train_acc  = history[:,2]
    val_loss   = history[:,3]
    val_acc    = history[:,4]

    # 損失(そんしつ)グラフ
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, train_loss, label=f"{model_name} - Train Loss")
    plt.plot(epochs, val_loss,   label=f"{model_name} - Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Loss")
    plt.legend()
    plt.show()

    # 精度(せいど)グラフ
    plt.figure(figsize=(9, 4))
    plt.plot(epochs, train_acc,  label=f"{model_name} - Train Acc")
    plt.plot(epochs, val_acc,    label=f"{model_name} - Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} Accuracy")
    plt.legend()
    plt.show()

# ===================================================
#  複数(ふくすう)モデルの学習履歴をまとめて比較(ひかく)してプロット
#   ・Validation Loss編
#   ・Validation Accuracy編
# ===================================================
def compare_histories(histories_dict):
    """
    histories_dict: { "model_name": history_ndarray, ... } の形で渡す。
      historyの各列は [epoch, train_loss, train_acc, val_loss, val_acc] を想定。
    """
    plt.figure(figsize=(10,6))
    for model_name, hist in histories_dict.items():
        plt.plot(hist[:,0], hist[:,3], label=f"{model_name} ValLoss")
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    for model_name, hist in histories_dict.items():
        plt.plot(hist[:,0], hist[:,4], label=f"{model_name} ValAcc")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

# ===================================================
#  画像(がぞう)とラベルを表示(ひょうじ)する関数
#   ・推定結果(すいていけっか)が正しければ黒、不正解なら赤色に
#   ・上部に "正解:推定" と表示、色分け
#   ・モデル名をタイトルに表示(任意)
# ===================================================
def show_images_labels(model_name, loader, classes, net, device):

    # 複数バッチから50枚の画像を収集
    images_list, labels_list = [], []
    for images, labels in loader:
        images_list.append(images)
        labels_list.append(labels)
        # 合計が50枚になったら終了
        if len(torch.cat(images_list)) >= 50:
            break

    # 50枚の画像を取り出す
    images = torch.cat(images_list)[:50]
    labels = torch.cat(labels_list)[:50]

    # モデルで予測
    net.eval()
    with torch.no_grad():
        inputs = images.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        predicted = torch.max(outputs, 1)[1]

    # グラフ表示
    plt.figure(figsize=(20, 15))
    plt.suptitle(f"Test Results - {model_name}", fontsize=24)  # モデル名を全体タイトルに表示
    for i in range(len(images)):
        ax = plt.subplot(5, 10, i + 1)
        label_idx = labels[i].item()
        pred_idx = predicted[i].item()

        label_name = classes[label_idx]
        predicted_name = classes[pred_idx]

        # 正解なら黒、不正解なら赤
        c = 'black' if label_idx == pred_idx else 'red'

        ax.set_title(f"{label_name}:{predicted_name}", color=c, fontsize=14)

        # Tensor → NumPy 変換
        img_np = images[i].cpu().numpy().transpose((1, 2, 0))
        img_np = (img_np + 1) / 2  # 正規化解除
        plt.imshow(img_np)
        ax.axis("off")
    plt.show()
