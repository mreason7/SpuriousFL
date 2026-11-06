"""Run only the three remaining paper datasets with the unified email protocol."""
import run_email_four as runner

runner.DEFAULT_DATASETS = runner.REMAINING_DATASETS
runner.BATCH_PREFIX = 'email-four'

if __name__ == '__main__':
    runner.main()
