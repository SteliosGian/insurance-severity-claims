name := "insurance_severity_claims"

version := "0.1"

scalaVersion := "2.11.8"

resolvers  += "MavenRepository" at "http://central.maven.org/maven2"

libraryDependencies += "org.apache.spark" %% "spark-core" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.1.0"
libraryDependencies += "org.apache.spark" %% "spark-mllib" % "2.1.0"