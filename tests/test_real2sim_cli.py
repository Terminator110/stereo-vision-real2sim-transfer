"""Offline, read-only launcher checks; run with `python -m unittest discover -s tests`."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'real2sim'
TAKE = 'take_20261104_193136_730622'


def call(*args):
    return subprocess.run([sys.executable, str(ENTRY), *args],
                          cwd=ROOT, capture_output=True, text=True)


class TestCLI(unittest.TestCase):
    def test_help(self):
        p = call('--help')
        self.assertEqual(p.returncode, 0, p.stderr)
        for item in ('record', 'process', 'annotate', 'replay', 'physics', 'doctor'):
            self.assertIn(item, p.stdout)

    def test_commands_dry_run(self):
        for command in ('record', 'process', 'annotate', 'replay', 'physics'):
            args = [command, '--dry-run']
            if command != 'record':
                args += ['--take', TAKE]
            p = call(*args)
            self.assertEqual(p.returncode, 0, f'{command}: {p.stdout}\n{p.stderr}')
            self.assertIn('nothing executed', p.stdout)

    def test_annotation_video_flag(self):
        # This test requires the real project's annotation source file.
        script = ROOT / "code" / "annotate_phases.py"
        if not script.exists():
            self.skipTest("annotation source unavailable in launcher-only package")
        p = call("annotate", "--take", TAKE, "--dry-run")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--video", p.stdout)
        self.assertIn("camera1.avi", p.stdout)
        self.assertNotIn(" --take ", p.stdout)

    def test_correct_calibrated_mapping(self):
        p = call('replay', '--take', TAKE, '--dry-run')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--table-yaw-deg -90', p.stdout)
        self.assertIn('--table-origin-world 0.385 -0.12 0.832', p.stdout)
        self.assertIn('--ik-mode xyz', p.stdout)

    def test_physics_uses_modern_trajectory(self):
        p = call('physics', '--take', TAKE, '--dry-run')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--trajectory', p.stdout)
        self.assertIn('real2sim_replay_' + TAKE + '/sim_qpos.npz', p.stdout)

    def test_no_mutation_on_doctor(self):
        p = call('doctor')
        self.assertIn('No project files changed', p.stdout)


if __name__ == '__main__':
    unittest.main()
