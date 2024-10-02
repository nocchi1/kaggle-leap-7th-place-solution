from abc import ABC, abstractmethod

class TrainerBase(ABC):
    @abstractmethod
    def train(self):
        """
        train_loaderのループを含む学習全体の処理
        """
        pass

    @abstractmethod
    def inference(self):
        """
        推論全体の処理
        """
        pass

    @abstractmethod
    def valid_predict(self):
        """
        trainメソッドの中で実行する関数
        valid_loaderの推論を行い, 予測値とスコアを返す
        """
        pass

    @abstractmethod
    def test_predict(self):
        """
        inferenceメソッドの中で実行する関数
        test_loaderの推論を行い, 予測値を返す
        """
        pass

    @abstractmethod
    def forward_step(self):
        """
        modelのforward処理と損失の計算を行う
        """
        pass

    @abstractmethod
    def _get_model(self):
        """
        モデルを取得する
        """
        pass

    @abstractmethod
    def _get_loss(self):
        """
        損失関数を取得する
        """
        pass

    @abstractmethod
    def _get_optimizer(self):
        """
        オプティマイザを取得する
        """
        pass

    @abstractmethod
    def _get_scheduler(self):
        """
        スケジューラを取得する
        """
        pass
