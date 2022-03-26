library(quantmod)
apple <- get(getSymbols("AAPL"))
data(ttrc)
dmi.adx <- ADX(ttrc[,c("High","Low","Close")])
HLC <- apple
try.xts(HLC, error=as.matrix)

"ADX" <-
  function(HLC, n=14) {
    
    # Welles Wilder's Directional Movement Index
    
    HLC <- try.xts(HLC, error=as.matrix)
    dH  <- momentum(HLC[,1])
    dL  <- -momentum(HLC[,2])
    
    DMIp <- ifelse( dH==dL | (dH< 0 & dL< 0), 0, ifelse( dH >dL, dH, 0 ) )
    DMIn <- ifelse( dH==dL | (dH< 0 & dL< 0), 0, ifelse( dH <dL, dL, 0 ) )
    
    TR    <- ATR(HLC)[,"tr"]
    TRsum <- wilderSum(TR, n=x)
    
    DIp <- 100 * wilderSum(DMIp, n=x) / TRsum
    DIn <- 100 * wilderSum(DMIn, n=x) / TRsum
    
    DX  <- 100 * ( abs(DIp - DIn) / (DIp + DIn) )
    
    maArgs <- list(n=x)
    
    # Default Welles Wilder EMA
    if(missing(maType)) {
      maType <- 'EMA'
      if(is.null(maArgs$wilder)) {
        # do not overwrite user-provided value
        maArgs$wilder <- TRUE
      }
    }
    
    ADX <- do.call( maType, c( list(DX), maArgs ) )
    
    result <- cbind( DIp, DIn, DX, ADX )
    colnames(result) <- c( "DIp", "DIn", "DX", "ADX" )
    
    reclass(result, HLC)
  }
ADX(apple, n= 14)
