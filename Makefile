deploy:
	sls prune -n 1
	sls deploy --verbose