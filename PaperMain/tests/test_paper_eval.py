"""Regression gates for frozen-policy OOD preprocessing and paired worlds."""
from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from paper_eval import make_configs, recipe_config
from calibration.profile import reject_calibration_episode_overlap

class PaperEvaluation(unittest.TestCase):
    def test_world_is_shared_by_all_three_recipes(self):
        for recipe in ('base','lrann','narrow'):
            env,policy = make_configs(recipe,'ID')
            self.assertEqual(env.seed,2024)
            self.assertEqual((env.ue_speed_min,env.ue_speed_max),(5.,40.))
            self.assertEqual((env.p_arrival_min,env.p_arrival_max),(.15,.5))
            self.assertEqual(policy.deadline_max,12)

    def test_d26_freezes_feature_normalization(self):
        for recipe in ('base','lrann','narrow'):
            env,policy = make_configs(recipe,'D26')
            self.assertEqual((env.deadline_min,env.deadline_max),(2,6))
            self.assertEqual(policy.deadline_max,12)
            self.assertIsNot(env,policy)

    def test_corrected_beta_applies_to_both_configs_without_changing_normalizers(self):
        profile = {'beta_rounded': [1., .8, .7, .6]}
        for recipe in ('base', 'lrann', 'narrow'):
            for world in ('ID', 'D26', 'K48'):
                env, policy = make_configs(recipe, world, calibration_profile=profile)
                self.assertEqual(env.la_beta_by_depth, (1., .8, .7, .6))
                self.assertEqual(policy.la_beta_by_depth, env.la_beta_by_depth)
                self.assertEqual(policy.deadline_max, 12)
                self.assertEqual(policy.ppo_batched_replay, recipe_config(recipe).ppo_batched_replay)
                self.assertEqual(recipe_config(recipe).la_beta_by_depth, (1.0018, .7499, .6592, .6058))
                self.assertEqual(policy.num_ue, env.num_ue)

    def test_final_evaluation_cannot_reuse_calibration_or_holdout_bands(self):
        profile = {'calibration': {'start': 50000, 'episodes': 36},
                   'holdout': {'start': 70000, 'episodes': 12}}
        for start, episodes in ((50000, 1), (49999, 2), (70011, 2)):
            with self.assertRaisesRegex(ValueError, 'overlaps the profile'):
                reject_calibration_episode_overlap(profile, start, episodes)
        for start, episodes in ((21000, 100), (49999, 1), (50036, 1), (70012, 100)):
            reject_calibration_episode_overlap(profile, start, episodes)

    def test_user_scaling_changes_shape_without_changing_feature_scales(self):
        for size in (8,48,60):
            env,policy = make_configs('narrow',f'K{size}')
            self.assertEqual(env.num_ue,size)
            self.assertEqual(policy.num_ue,size)
            for name in ('deadline_max','cqi_norm_const','b_norm','queue_size','age_norm_max'):
                self.assertEqual(getattr(policy,name),getattr(recipe_config('narrow'),name))

if __name__ == '__main__': unittest.main()
