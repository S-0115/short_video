import torch

class HiddenRecoder():
    # for evaluate a policy

    def __init__(self, device, encoder, history_length):
        self.device = device
        self.encoder = encoder
        self.encoder.to(self.device)
        self.history_length = history_length
        # dict of dict, rec[reset_after][up_to] = [curr_latent_mean, curr_latent_logvar, hidden_state]
        self.rec = {}
        self.step = 0

    def _add_record(self, latent_mean, latent_logvar, hidden, reset_after, up_to):
        assert latent_mean.dim() == 2
        assert latent_logvar.dim() == 2
        if reset_after not in self.rec.keys():
            self.rec[reset_after] = {
                up_to: [latent_mean.clone().detach(),
                        latent_logvar.clone().detach(),
                        hidden]}
        else:
            self.rec[reset_after][up_to] = [latent_mean.clone().detach(),
                                            latent_logvar.clone().detach(),
                                            hidden]

    def get_record(self, reset_after, up_to, label):
        # label can be 'hidden' or 'latent'
        assert reset_after is not None
        try:
            if label == 'hidden':
                return self.rec[reset_after][up_to][-1]
            elif label == 'latent':
                return self.rec[reset_after][up_to][0], self.rec[reset_after][up_to][1]
            else:
                raise ValueError('label can only be "hidden" or "latent"')
        except ValueError:
            raise ValueError("No record found")

    def encoder_init(self, step):
        curr_latent_mean, curr_latent_logvar, hidden_state = self.encoder.prior(1)
        curr_latent_mean = curr_latent_mean.squeeze(0)
        curr_latent_logvar = curr_latent_logvar.squeeze(0)
        if type(hidden_state) == tuple:
            hidden_state = (hidden_state[0].squeeze(0), hidden_state[1].squeeze(0))
        elif type(hidden_state) == torch.Tensor:
            hidden_state = hidden_state.squeeze(0)
        elif type(hidden_state) == list:
            hidden_state = hidden_state
        # reset after == current_step 的时候，这个 hidden_state 应该就是 encoder.prior
        self._add_record(curr_latent_mean,
                         curr_latent_logvar, hidden_state, reset_after=step, up_to=step)
        return curr_latent_mean, curr_latent_logvar

    def encoder_step(self, state):
        # assume in this line self.step = 3, and the input action state rew come after step 4
        # best_reset_after takes value in [0,1,2,3,4]
        # for i in range(self.step + 1):  # i in [0,1,2,3]
        for i in range(max(self.step + 1 - self.history_length, 0), self.step + 1):  # i in [0,1,2,3]
            # print(i)
            # extend hidden state recording, for recordings reset after 0, 1, 2, 3
            prev_hidden_state = self.get_record(
                reset_after=i, up_to=self.step, label='hidden')
            curr_latent_mean, curr_latent_logvar, hidden_state = self.encoder(state.to(self.device), prev_hidden_state)
            self._add_record(curr_latent_mean,curr_latent_logvar, hidden_state, reset_after=i, up_to=self.step+1)

        # create hidden_state for reset after 4 and up to 4
        self.encoder_init(step=self.step + 1)
        self.step += 1
        # print()
        if self.step - self.history_length > 0:
            # print('del', self.step - 5)
            del self.rec[self.step - self.history_length - 1]

