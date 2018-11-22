# Compute Feature Representation for Images


## DataFrames w/ Bounding Boxes

The following dataframes (created by `../Preproc/preproc.py`) can be input to the extraction:

* `saiapr_bbdf mscoco_bbdf cocogrprops_bbdf vgregdf vgobjdf flickr_bbdf` 

(Can be copied over like this to script, e.g.:

```sh
python extract.py -c ../Config/geforce.cfg vgg19-fc2 LIST
```

)

Any of these can serve as basis for extracting features for the whole image rather than for regions of it.
